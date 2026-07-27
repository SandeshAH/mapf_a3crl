import numpy as np
from collections import OrderedDict
from threading import Lock
import sys
import random
import math
import copy
sys.path.append('../')
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
import torch
import torch.multiprocessing as mp
from shared_optim import SharedAdam
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import multiprocessing
import scipy.signal as signal
import gc
import heapq

class testMAPFEnv():
    

    # Initialize env
    def __init__(self, num_agents=1, observation_size=10, DIAGONAL_MOVEMENT=False, SIZE=20, PROB=.1):
        
        # Initialize variables
        self.num_agents        = num_agents
        
        # 1 - obstacles
        # 0 - free
        self.observation_size  = torch.tensor(observation_size)
        self.SIZE              = torch.tensor(SIZE)
        self.PROB              = torch.tensor(PROB)
        self.fresh             = torch.tensor(True)
        
        self.finished          = torch.tensor(False)
        self.DIAGONAL_MOVEMENT = torch.tensor(DIAGONAL_MOVEMENT)

        # Define shared tensors, initialized with zeros
        self.start_pos = torch.zeros((self.num_agents, 2))
        self.goal_pos = torch.zeros((self.num_agents, 2))
        self.gridworld = torch.zeros((self.SIZE, self.SIZE), dtype=torch.int)
        self.obstacle_world = torch.zeros((self.SIZE, self.SIZE), dtype=torch.int)
        self.goalworld = torch.zeros((self.SIZE, self.SIZE), dtype=torch.int)
        self.agents_past = torch.zeros((self.num_agents, 2))
        
        # # Generate world and assign start and end pos for agents
        self._reset_world()
        self.blocking = torch.tensor(False)
        self.opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
        
        self.dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
        self.actionDict={v:k for k,v in self.dirDict.items()}
        self.ACTION_COST, self.IDLE_COST, self.GOAL_REWARD, self.COLLISION_REWARD,self.FINISH_REWARD,self.BLOCKING_COST = -0.3, -0.5, 0.0, -2.0,20.0,-1.0
        

        
    def get_pos(self,agent_id):
        # to get the current position of the current agent in the world
        
        cur_pos=torch.stack(torch.where(self.gridworld==agent_id)).T
        if not len(cur_pos):
            print(f'agent id ={agent_id} position not found{self.start_pos[agent_id-2]}',flush=True)
            return torch.tensor([-1,-1])
        return cur_pos[0]
        
    def get_goal(self,agent_id):
        # to get the goal position of the current agent
        return self.goal_pos[agent_id-2] #-2 because started agentid  from 2 
    def get_agents_past(self,agent_id):
        # to get the goal position of the current agent
        return self.agents_past[agent_id-2] #-2 because started agentid  from 2 
    def getDir(self,action):
        #get the direction for the curresponding action from the  dictionary
        # 0     1  2  3  4 
        # still N  E  S  W
        return self.dirDict[action]
    
    
    def act(self, action, agent_id):
        # try to execture action and return whether action was executed or not and why
        #returns:
        #     2: action executed and left goal
        #     1: action executed and reached goal (or stayed on)
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with wall
        #    -3: collision with robot

      
        direction = self.getDir(action)
        moved = self.moveAgent(direction,agent_id)
        return moved
    
    def moveAgent(self, direction, agent_id):
        
        #get current position of the agent
        ax=self.get_pos(agent_id)[0]
        ay=self.get_pos(agent_id)[1]
        dx,dy =direction[0], direction[1]
        #let's look at the validity of the move
        if(ax+dx>=self.gridworld.shape[0] or ax+dx<0 or ay+dy>=self.gridworld.shape[1] or ay+dy<0):
            # print(agent_id,"agent out of bounds")
            #out of bounds
            return -1
            
        if(self.gridworld[ax+dx,ay+dy]==1):#collide with static obstacle
            # print(agent_id,"collide with static obstacle")
            return -2
            
        if(self.gridworld[ax+dx,ay+dy]>1 and self.gridworld[ax+dx,ay+dy]!=agent_id):#collide with robot
            # print(agent_id,"collide with robot",self.gridworld[ax+dx,ay+dy])
            return -3
        
        #carry out the action
        self.gridworld[ax,ay] = 0
        self.gridworld[ax+dx,ay+dy] = agent_id #moved the agent curresponding to the action
        
        return 0
        
    
     # Returns an observation of an agent
    
    def _observe1(self,agent_id):
        assert agent_id > 1 # because started agent_id  from 2 
        # print(agent_id,"obseravtion start",flush=True)

        # Get agent's position
        position = self.get_pos(agent_id)
        half_obs_size = torch.div(self.observation_size, 2, rounding_mode='floor')
        
        # Calculate the top-left corner of the observation window
        top_left=(position[0]-half_obs_size,position[1]-half_obs_size)
        
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        obs_shape=(self.observation_size,self.observation_size) # 10x10 observation shape
        
        # Initialize the obstacles map of the current agent in the observation view
        obs_map              = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # Initialize the map of the position of all agents in the observation view
        poss_map             = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # Initialize the map of the goal of the current agent in the observation view
        goal_map             = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # Initialize the map of the goals of the other visible agents in the observation view
        goalss_map             = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # print(agent_id,"maps initialized",flush=True)
        #For storing details of all visible agents in the current agent's observation view
        visible_agents=[]
        for i in range(top_left[0],top_left[0]+self.observation_size):
            for j in range(top_left[1],top_left[1]+self.observation_size):
                if i>=self.gridworld.shape[0] or i<0 or j>=self.gridworld.shape[1] or j<0:
                    #out of bounds, just treat as an obstacle
                    obs_map[i-top_left[0],j-top_left[1]]=1
                    continue
                
                if self.gridworld[i,j]==1:
                    #obstacles
                    obs_map[i-top_left[0],j-top_left[1]]=1
                
                if torch.equal(torch.tensor([i,j]), self.get_pos(agent_id)):
                    #marked current agents' positions in poss_map
                    poss_map[i-top_left[0],j-top_left[1]]=1
                
                if self.gridworld[i,j]>1 and self.gridworld[i,j]!=agent_id:
                    #stored details of all visible agents in the current agent's observation view
                    visible_agents.append(self.gridworld[i,j])
                    #marked other agents' positions in poss_map
                    poss_map[i-top_left[0],j-top_left[1]]=1
                    #consider other agent as obtacle in the osbtacle map
                    # obs_map[i-top_left[0],j-top_left[1]]=1
                
                if torch.equal(torch.tensor([i,j]), self.get_goal(agent_id)):
                    #marked current agent goal in Goal_map
                    goal_map[i-top_left[0],j-top_left[1]]=1
        
         #if the goal pos of visible agents outside the observation view we clamp it           
        for agent in visible_agents:
            x, y = self.get_goal(agent)  # Assuming x and y are tensors

            # Clamp the goal of visible agents to fit in the observation window
            min_node_x = torch.max(top_left[0], torch.min(top_left[0] + self.observation_size - 1, x))
            min_node_y = torch.max(top_left[1], torch.min(top_left[1] + self.observation_size - 1, y))
            min_node = (min_node_x, min_node_y)

            # Update the goals_map 
            goalss_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

          

         
        dx=self.get_goal(agent_id)[0]-self.get_pos(agent_id)[0]
        dy=self.get_goal(agent_id)[1]-self.get_pos(agent_id)[1]
        mag = torch.sqrt(dx.cpu()**2 + dy.cpu()**2)  # Compute the magnitude

        # Avoid division by zero by using torch.where
        # Only divide by mag where mag is not zero
        dx = torch.where(mag != 0, dx / mag, torch.zeros_like(dx))
        dy = torch.where(mag != 0, dy / mag, torch.zeros_like(dy))

       
        input_to_nn=torch.stack([poss_map, goal_map, goalss_map, obs_map], dim=0)
        goal_pos_to_nn = torch.tensor([dx, dy,mag], dtype=torch.float32)
        
        return input_to_nn,goal_pos_to_nn
    def _observe(self,agent_id):
        assert agent_id > 1 # because started agent_id  from 2 
        # print(agent_id,"obseravtion start",flush=True)

        # Get agent's position
        position = self.get_pos(agent_id)
        half_obs_size = torch.div(self.observation_size, 2, rounding_mode='floor')
        
        # Calculate the top-left corner of the observation window
        top_left=(position[0]-half_obs_size,position[1]-half_obs_size)
        
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        obs_shape=(self.observation_size,self.observation_size) # 10x10 observation shape
        
        # Initialize the obstacles map of the current agent in the observation view
        obs_map              = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # Initialize the map of the position of all agents in the observation view
        poss_map             = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # Initialize the map of the goal of the current agent in the observation view
        goal_map             = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # Initialize the map of the goals of the other visible agents in the observation view
        goalss_map             = torch.zeros(obs_shape,dtype=torch.float32,device='cpu')
        # print(agent_id,"maps initialized",flush=True)
        #For storing details of all visible agents in the current agent's observation view
        visible_agents=[]
        for i in range(top_left[0],top_left[0]+self.observation_size):
            for j in range(top_left[1],top_left[1]+self.observation_size):
                if i>=self.gridworld.shape[0] or i<0 or j>=self.gridworld.shape[1] or j<0:
                    #out of bounds, just treat as an obstacle
                    obs_map[i-top_left[0],j-top_left[1]]=1
                    continue
                
                if self.gridworld[i,j]==1:
                    #obstacles
                    obs_map[i-top_left[0],j-top_left[1]]=1
                
                # if torch.equal(torch.tensor([i,j]), self.get_pos(agent_id)):
                if (i, j) == tuple(self.get_pos(agent_id).tolist()):
                        
                    #marked current agents' positions in poss_map
                    poss_map[i-top_left[0],j-top_left[1]]=1
                
                if self.gridworld[i,j]>1 and self.gridworld[i,j]!=agent_id:
                    #stored details of all visible agents in the current agent's observation view
                    visible_agents.append(self.gridworld[i,j])
                    #marked other agents' positions in poss_map
                    poss_map[i-top_left[0],j-top_left[1]]=1
                    
                
                if (i, j) == tuple(self.get_goal(agent_id).tolist()):
                    #marked current agent goal in Goal_map
                    goal_map[i-top_left[0],j-top_left[1]]=1
        
         #if the goal pos of visible agents outside the observation view we clamp it           
        for agent in visible_agents:
            x, y = self.get_goal(agent)  # Assuming x and y are tensors

            # Clamp the goal of visible agents to fit in the observation window
            min_node_x = torch.max(top_left[0], torch.min(top_left[0] + self.observation_size - 1, x))
            min_node_y = torch.max(top_left[1], torch.min(top_left[1] + self.observation_size - 1, y))
            min_node = (min_node_x, min_node_y)

            # Update the goals_map 
            goalss_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

          

         
        dx=self.get_goal(agent_id)[0]-self.get_pos(agent_id)[0]
        
        dy=self.get_goal(agent_id)[1]-self.get_pos(agent_id)[1]
       

       
        
          
        
        
        mag = torch.sqrt(dx.cpu()**2 + dy.cpu()**2)  # Compute the magnitude

        # Avoid division by zero by using torch.where
        # Only divide by mag where mag is not zero
        dx = torch.where(mag != 0, dx / mag, torch.zeros_like(dx))
        dy = torch.where(mag != 0, dy / mag, torch.zeros_like(dy))

       
        input_to_nn=torch.stack([poss_map,obs_map, goal_map, goalss_map], dim=0)
        # input_to_nn=torch.stack([poss_map,obs_map,goal_map], dim=0)
        goal_pos_to_nn = torch.tensor([dx, dy,mag], dtype=torch.float32)
        
        return input_to_nn,goal_pos_to_nn
    
    def _listNextValidActions(self, agent_id, prev_action=0,episode=0):
        #Give a list of next possible valid action for the current agents
        available_actions = [0] # staying still always allowed

        # Get current agent position
        agent_pos = self.get_pos(agent_id)
        ax,ay     = agent_pos[0],agent_pos[1]
        n_moves   = 9 if self.DIAGONAL_MOVEMENT else 5

        for action in range(1,n_moves):
            direction = self.getDir(action)
            dx,dy     = direction[0],direction[1]
            # Calculate the new position if the action is taken (ax+dx,ay+dy)
            if(ax+dx>=self.gridworld.shape[0] or ax+dx<0 or ay+dy>=self.gridworld.shape[1] or ay+dy<0):
                #out of bounds
                continue
            if(self.gridworld[ax+dx,ay+dy]==1):#collide with static obstacle
                continue
            if(self.gridworld[ax+dx,ay+dy]>0):#collide with robot
                continue
                   
            #otherwise we are ok to carry out the action
            available_actions.append(action)

                
        return available_actions
    
    def _step1(self, action_input,episode=0):
        self.fresh = False
        n_actions = 9 if self.DIAGONAL_MOVEMENT else 5

        # Check action input
        assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
        assert action_input[1] in range(n_actions), 'Invalid action'
        assert action_input[0] in range(2, self.num_agents+2) #2 bcz agent_id start from 2

        # Parse action input
        agent_id = action_input[0]
        action   = action_input[1]
        
        action_status = self.act(action,agent_id)
        valid_action = action_status >=0 #for imitation learning
        
        
        self.blocking.fill_(False)
        self.blocking.fill_(False)
        
        if action_status==0:#moved successfully
            if action==0:
                reward=self.IDLE_COST
            else:
                reward=self.ACTION_COST
        else:
            reward=self.COLLISION_REWARD
            
    
        # Perform observation after action taken
        new_state = self._observe(agent_id) 
        # Done?
        done = self._done() #check all agent reached goal?
        self.finished |= done

        # next valid actions
        nextActions = self._listNextValidActions(agent_id, action,episode=episode)

        # on_goal estimation
        on_goal = torch.equal(self.get_pos(agent_id),self.get_goal(agent_id))
        
        if on_goal:
            reward=self.GOAL_REWARD
        if self.finished:
            reward=self.FINISH_REWARD
        
        return reward,nextActions,valid_action
    
    def _step(self, action_input,episode=0):
        self.fresh = False
        n_actions = 9 if self.DIAGONAL_MOVEMENT else 5

        # Check action input
        assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
        assert action_input[1] in range(n_actions), 'Invalid action'
        assert action_input[0] in range(2, self.num_agents+2) #2 bcz agent_id start from 2

        # Parse action input
        agent_id = action_input[0]
        action   = action_input[1]
        
        # Execute action & determine reward
        action_status = self.act(action,agent_id)
        valid_action = action_status >=0 #for imitation learning
        
        
        # self.blocking.fill_(False)
        self.blocking = torch.tensor(False)
        if action==0:#staying still
           
            if action_status == 1:#stayed on goal
                reward=self.GOAL_REWARD
                x=self._get_blocking_reward(agent_id) #check agent blocking other agents
                # print(agent_id,"before adding blocking pos",(self.get_pos(agent_id)),"goal",(self.get_goal(agent_id)),x)
                 #blocking reward is negative so add it with goal reward
                if x<0:
                    # self.blocking.fill_(True)
                    self.blocking = torch.tensor(True)
                    reward=x
                    # print(agent_id,"after adding blocking",reward,x)
                else:
                    reward=self.GOAL_REWARD

            elif action_status == 0:#stayed off goal
                reward=self.IDLE_COST
        else:#moving
            if (action_status == 1): # reached goal
                reward = self.GOAL_REWARD
            elif (action_status == -3 or action_status==-2 or action_status==-1): # collision
                reward = self.COLLISION_REWARD
            elif (action_status == 2): #left goal
                reward=self.ACTION_COST
            else:
                reward=self.ACTION_COST
    
        # Perform observation after action taken
        new_state = self._observe(agent_id) 

        
        
        
        # Done?
        done = self._done() #check all agent reached goal?
        self.finished |= done
        
        if self.finished:
            reward=self.FINISH_REWARD #All agents successfully reached goal give reward 
        

        # next valid actions
        nextActions = self._listNextValidActions(agent_id, action,episode=episode)

        # on_goal estimation
        on_goal=False
        on_goal = torch.equal(self.get_pos(agent_id),self.get_goal(agent_id))
        
        return new_state,reward,nextActions,on_goal,self.blocking,valid_action
    

    def _is_world_valid(self,world,start,goal):
        """Check if a path exists between start and goal in a grid world using A*"""
        if world[start[0], start[1]] == 1 or world[goal[0], goal[1]] == 1:
            # Start or goal is blocked
            return False

        def heuristic(a, b):
            """Heuristic function: Manhattan distance between points a and b."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Priority queue for open set
        open_set = []
        heapq.heappush(open_set, (0, start))  # (priority, (x, y))

        # Cost from start to each cell
        g_score = {start: 0}

        # Track visited nodes
        visited = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)

            # Check if we reached the goal
            if current == goal:
                return True

            # Explore neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < world.shape[0] and 0 <= neighbor[1] < world.shape[1]:  # Use .shape for NumPy arrays
                    if world[neighbor[0], neighbor[1]] == 1:
                        # Skip obstacles
                        continue

                    tentative_g_score = g_score[current] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
        # If we exhaust the open set without finding the goal
        return False
    
    def _reset_world(self):
        
        valid_world=False
        while not valid_world:
            self.finished.fill_(False)
            # Generate all possible positions in the grid
            all_positions = torch.cartesian_prod(torch.arange(self.SIZE), torch.arange(self.SIZE))
            all_positions = all_positions[torch.randperm(all_positions.size(0))]

            # Update the starting positions in-place
            self.start_pos.zero_()  # Clear previous values if needed
            self.start_pos.copy_(all_positions[:self.num_agents])  # Copy new positions into existing shared tensor

            # Update remaining positions and select goal positions
            remaining_positions = all_positions[self.num_agents:]
            remaining_positions = remaining_positions[torch.randperm(remaining_positions.size(0))]

            # Update goal positions in-place
            self.goal_pos.zero_()  # Clear previous values if needed
            self.goal_pos.copy_(remaining_positions[:self.num_agents])  # Copy new positions into existing shared tensor

            # Ensure start_pos and goal_pos are of type Long for indexing
            self.start_pos = self.start_pos.long()
            self.goal_pos = self.goal_pos.long()

            # Update gridworld, ensuring changes are in-place
            self.gridworld.zero_()  # Clear previous world setup
            self.gridworld.copy_((torch.rand(self.SIZE, self.SIZE) < self.PROB).to(torch.int))

            # Assign start and goal positions as free in the grid
            self.gridworld[self.start_pos[:, 0], self.start_pos[:, 1]] = 0
            self.gridworld[self.goal_pos[:, 0], self.goal_pos[:, 1]] = 0

            # Update obstacle world in-place
            self.obstacle_world.zero_()  # Clear previous obstacle world setup
            self.obstacle_world.copy_((self.gridworld >= 1).int())
            # Validate paths
            valid_world = True
            for i in range(self.num_agents):
                obstcl_world0=self.obstacle_world.cpu().numpy()
                start = tuple(self.start_pos[i].tolist())
                goal = tuple(self.goal_pos[i].tolist())
                if not self._is_world_valid(obstcl_world0, start, goal):
                    # print("world created is not valid")
                    valid_world = False
                    break
            # Define the goal world in-place
            self.goalworld.zero_()  # Clear previous goal world setup
            self.goalworld.copy_(self.gridworld)  # Clone to ensure size match with gridworld

            # Update agents' past positions in-place
            self.agents_past.zero_()  # Clear previous positions
            self.agents_past.copy_(self.start_pos)

            # Update gridworld with agent and goal positions
            agent_counter = 2  # Start ID from 2 because 1 represents obstacles
            agent_goal_counter = 2 # Start ID from 2 because 1 represents obstacles
            for pos in self.start_pos:
                x, y = pos.tolist()
                self.gridworld[x, y] = agent_counter
                agent_counter += 1

            for gol in self.goal_pos:
                x, y = gol.tolist()
                self.goalworld[x, y] = agent_goal_counter
                agent_goal_counter += 1
    
    def _done(self):
        "To check all agents reach its goal"
        numComplete = 0
        for agent in range(2,self.num_agents+2):
            agent_pos = self.get_pos(agent)
            if self.goalworld[agent_pos[0],agent_pos[1]].item() == agent:
                numComplete += 1
                
        return numComplete==self.num_agents
    
 

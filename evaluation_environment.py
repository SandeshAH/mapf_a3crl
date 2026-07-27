import numpy as np
from collections import OrderedDict

import sys
import random
import os
import copy
sys.path.append('../')
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
import torch
import sys
import scipy.signal as signal
import heapq
import pickle
class MAPFEnv():
    

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
        self.ACTION_COST, self.IDLE_COST, self.GOAL_REWARD, self.COLLISION_REWARD,self.FINISH_REWARD,self.BLOCKING_COST = -0.3, -.5, 0.1, -2.,20.,-0.01
        


    def export_env(self, filename='env_state.pkl'):
        """
        Save everything needed to recreate the world exactly.
        """
        data = {
            'num_agents': self.num_agents.item() if isinstance(self.num_agents, torch.Tensor) else self.num_agents,
            'SIZE': self.SIZE.item() if isinstance(self.SIZE, torch.Tensor) else self.SIZE,
            'PROB': self.PROB.item() if isinstance(self.PROB, torch.Tensor) else self.PROB,
            'observation_size': self.observation_size.item() if isinstance(self.observation_size, torch.Tensor) else self.observation_size,
            'DIAGONAL_MOVEMENT': bool(self.DIAGONAL_MOVEMENT),
            'gridworld': self.gridworld.clone().cpu(),
            'obstacle_world': self.obstacle_world.clone().cpu(),
            'start_pos': self.start_pos.clone().cpu(),
            'goal_pos': self.goal_pos.clone().cpu(),
            'goalworld': self.goalworld.clone().cpu()
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Environment exported to {filename}")

    def import_env(self, filename='env_state.pkl'):
        """
        Load and recreate the environment exactly as saved.
        SAFE version:
        - No tensor slicing used as indices
        - All indices converted to Python int
        - Compatible with A3C multiprocessing
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # --------------------------------------------------
        # Restore scalar parameters (NOT shared tensors here)
        # --------------------------------------------------
        self.num_agents = int(data['num_agents'])
        self.SIZE = int(data['SIZE'])
        self.PROB = float(data['PROB'])
        self.observation_size = int(data['observation_size'])
        self.DIAGONAL_MOVEMENT = bool(data['DIAGONAL_MOVEMENT'])
        self.finished = False

        N = self.num_agents
        SIZE = self.SIZE

        # --------------------------------------------------
        # Restore world tensors (force CPU + long where needed)
        # --------------------------------------------------
        self.gridworld = data['gridworld'].clone().cpu().int()
        self.obstacle_world = data['obstacle_world'].clone().cpu().int()

        self.start_pos = data['start_pos'].clone().cpu().long()
        self.goal_pos = data['goal_pos'].clone().cpu().long()

        self.goalworld = data['goalworld'].clone().cpu().int()

        # Agents past = start positions
        self.agents_past = self.start_pos.clone()

        # --------------------------------------------------
        # SAFELY clear any agent markers from gridworld
        # --------------------------------------------------
        for i in range(self.start_pos.shape[0]):
            sx = int(self.start_pos[i, 0].item())
            sy = int(self.start_pos[i, 1].item())
            self.gridworld[sx, sy] = 0

        for i in range(self.goal_pos.shape[0]):
            gx = int(self.goal_pos[i, 0].item())
            gy = int(self.goal_pos[i, 1].item())
            self.gridworld[gx, gy] = 0

        # --------------------------------------------------
        # Re-place agents into gridworld (IDs 2..N+1)
        # --------------------------------------------------
        agent_id = 2
        for i in range(N):
            x = int(self.start_pos[i, 0].item())
            y = int(self.start_pos[i, 1].item())
            self.gridworld[x, y] = agent_id
            agent_id += 1

        # --------------------------------------------------
        # Re-place goals into goalworld (IDs 2..N+1)
        # --------------------------------------------------
        agent_id = 2
        for i in range(N):
            x = int(self.goal_pos[i, 0].item())
            y = int(self.goal_pos[i, 1].item())
            self.goalworld[x, y] = agent_id
            agent_id += 1

        print(f"✅ Environment safely imported from {filename}")

    def import_env1(self, filename='env_state.pkl'):
        """
        Load and recreate the environment exactly as saved.
        This version is simplified for evaluation (non-shared tensors).
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Directly assign values (not shared tensors)
        self.num_agents = data['num_agents']
        self.SIZE = data['SIZE']
        self.PROB = data['PROB']
        self.observation_size = data['observation_size']
        self.DIAGONAL_MOVEMENT = data['DIAGONAL_MOVEMENT']
        self.finished = False

        # Replace existing tensors with loaded versions
        self.gridworld = data['gridworld']
        self.obstacle_world = data['obstacle_world']
        self.start_pos = data['start_pos']
        self.goal_pos = data['goal_pos']
        self.goalworld = data['goalworld']
        self.agents_past = self.start_pos.clone()

        # Place agent IDs back in gridworld
        self.gridworld[self.start_pos[:, 0], self.start_pos[:, 1]] = 0
        self.gridworld[self.goal_pos[:, 0], self.goal_pos[:, 1]] = 0

        agent_id = 2
        for pos in self.start_pos:
            x, y = pos.tolist()
            self.gridworld[x, y] = agent_id
            agent_id += 1

        agent_id = 2
        for pos in self.goal_pos:
            x, y = pos.tolist()
            self.goalworld[x, y] = agent_id
            agent_id += 1

        print(f"✅ Loaded environment from {filename}")
            
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
    
    def moveAgent1(self, direction, agent_id):
        
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
    def moveAgent(self, direction, agent_id):
        #try to move agent and return the status

        #get current position of the agent
        ax=self.get_pos(agent_id)[0]
        ay=self.get_pos(agent_id)[1]

        # Not moving is always allowed
        if(direction==(0,0)):
            self.agents_past[agent_id-2]=self.get_pos(agent_id)
            return 1 if self.goalworld[ax,ay]==agent_id else 0

        # Otherwise, let's look at the validity of the move
        dx,dy =direction[0], direction[1]
        if(ax+dx>=self.gridworld.shape[0] or ax+dx<0 or ay+dy>=self.gridworld.shape[1] or ay+dy<0):
            #out of bounds
            return -1
        if(self.gridworld[ax+dx,ay+dy]==1):#collide with static obstacle
            return -2
        if(self.gridworld[ax+dx,ay+dy]>1 and self.gridworld[ax+dx,ay+dy]!=agent_id):#collide with robot
            # print("collided with robot")
            return -3
        
        # No collision: we can carry out the action
        self.gridworld[ax,ay] = 0
        self.gridworld[ax+dx,ay+dy] = agent_id #moved the agent curresponding to the action
        self.agents_past[agent_id-2]=self.get_pos(agent_id)#updated the past
        # self.agents[agent_id-1] = (ax+dx,ay+dy)
        if self.goalworld[ax+dx,ay+dy]==agent_id:
            return 1 #action executed and reached goal
        
        elif self.goalworld[ax+dx,ay+dy]!=agent_id and self.goalworld[ax,ay]==agent_id:
            return 2 #action executed and left goal
        
        else:
            return 0 #action executed
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
                
                if torch.equal(torch.tensor([i,j]), self.get_pos(agent_id)):
                    #marked current agents' positions in poss_map
                    poss_map[i-top_left[0],j-top_left[1]]=1
                
                if self.gridworld[i,j]>1 and self.gridworld[i,j]!=agent_id:
                    #stored details of all visible agents in the current agent's observation view
                    visible_agents.append(self.gridworld[i,j])
                    #marked other agents' positions in poss_map
                    poss_map[i-top_left[0],j-top_left[1]]=1
                    
                
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

       
        input_to_nn=torch.stack([poss_map,obs_map, goal_map, goalss_map], dim=0)
        # input_to_nn=torch.stack([poss_map, obs_map, goal_map], dim=0)
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
        robot_collision=False
        obstacle_collision=False
        
        # self.blocking.fill_(False)
        self.blocking = torch.tensor(False)
        if action==0:#staying still
           
            if action_status == 1:#stayed on goal
                reward=self.GOAL_REWARD
                x=self._get_blocking_reward(agent_id) #check agent blocking other agents
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
                if action_status == -3:
                   robot_collision=True 
                else:
                    obstacle_collision=True
                reward = self.COLLISION_REWARD
            elif (action_status == 2): #left goal
                reward=self.ACTION_COST
            else:
                reward=self.ACTION_COST
    
        # Perform observation after action taken
        new_state = self._observe(agent_id) 

        
        # on_goal estimation
        on_goal=False
        on_goal = torch.equal(self.get_pos(agent_id),self.get_goal(agent_id))

        # episode Done?
        done = self._done() #check all agent reached goal?
        self.finished |= done
        
        # if self.finished:
        #     reward=self.FINISH_REWARD #All agents successfully reached goal give reward 20
        

        # next valid actions
        nextActions = self._listNextValidActions(agent_id, action,episode=episode)

        
       
        
        return new_state,reward,nextActions,on_goal,self.blocking,valid_action,robot_collision,obstacle_collision
    

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
    
    def _structured_world_1(self):
        """
        Evaluation-only structured world (ROBUST VERSION)

        Properties:
        - Horizontal corridor in the middle
        - Left & right bands same width as corridor
        - Starts ONLY near corners (top/bottom of bands)
        - Goals on diagonally opposite corners
        - Start positions sampled WITHOUT replacement
        - Capacity checked explicitly
        """

        # --------------------------------------------------
        # Safe value extraction (int or tensor)
        # --------------------------------------------------
        SIZE = self.SIZE if isinstance(self.SIZE, int) else int(self.SIZE.item())
        N = self.num_agents if isinstance(self.num_agents, int) else int(self.num_agents.item())

        self.finished = False

        # --------------------------------------------------
        # PARAMETERS (tune here if needed)
        # --------------------------------------------------
        BASE_CORRIDOR_FRAC = 0.20   # % of grid height
        EXTRA_WIDTH = 2             # widen corridor

        corridor_width = max(5, int(SIZE * BASE_CORRIDOR_FRAC) + EXTRA_WIDTH)
        band_width = corridor_width

        corridor_y0 = SIZE // 2 - corridor_width // 2
        corridor_y1 = corridor_y0 + corridor_width

        left_x0, left_x1 = 0, band_width
        right_x0, right_x1 = SIZE - band_width, SIZE

        corner_h = band_width  # corner height = band width

        # --------------------------------------------------
        # 1. Initialize full obstacle grid
        # --------------------------------------------------
        self.gridworld.zero_()
        self.gridworld.fill_(1)

        # --------------------------------------------------
        # 2. Free horizontal corridor
        # --------------------------------------------------
        self.gridworld[corridor_y0:corridor_y1, :] = 0

        # --------------------------------------------------
        # 3. Free left & right bands
        # --------------------------------------------------
        self.gridworld[:, left_x0:left_x1] = 0
        self.gridworld[:, right_x0:right_x1] = 0

        # --------------------------------------------------
        # 4. Build ALL valid start cells (corners only)
        # --------------------------------------------------
        start_cells = []

        # top-left & top-right corners
        for r in range(0, corner_h):
            for c in range(left_x0, left_x1):
                start_cells.append((r, c))
            for c in range(right_x0, right_x1):
                start_cells.append((r, c))

        # bottom-left & bottom-right corners
        for r in range(SIZE - corner_h, SIZE):
            for c in range(left_x0, left_x1):
                start_cells.append((r, c))
            for c in range(right_x0, right_x1):
                start_cells.append((r, c))

        start_cells = torch.tensor(start_cells, dtype=torch.long)

        # --------------------------------------------------
        # 5. Capacity check (CRITICAL)
        # --------------------------------------------------
        max_agents = start_cells.shape[0]
        if N > max_agents:
            raise ValueError(
                f"Too many agents ({N}) for this world.\n"
                f"Max allowed = {max_agents} "
                f"(corridor_width={corridor_width}, SIZE={SIZE})"
            )

        # --------------------------------------------------
        # 6. Sample start positions WITHOUT replacement
        # --------------------------------------------------
        perm = torch.randperm(max_agents)[:N]
        self.start_pos.zero_()
        self.start_pos.copy_(start_cells[perm])

        # --------------------------------------------------
        # 7. Assign goals (diagonal opposite)
        # --------------------------------------------------
        self.goal_pos.zero_()
        for i in range(N):
            r, c = self.start_pos[i]
            self.goal_pos[i] = torch.tensor([SIZE - 1 - r, SIZE - 1 - c])

        self.start_pos = self.start_pos.long()
        self.goal_pos = self.goal_pos.long()

        # --------------------------------------------------
        # 8. Obstacle world
        # --------------------------------------------------
        self.obstacle_world.zero_()
        self.obstacle_world.copy_((self.gridworld == 1).int())

        # --------------------------------------------------
        # 9. Goal world
        # --------------------------------------------------
        self.goalworld.zero_()
        self.goalworld.copy_(self.gridworld)

        # --------------------------------------------------
        # 10. Agents past
        # --------------------------------------------------
        self.agents_past.zero_()
        self.agents_past.copy_(self.start_pos)

        # --------------------------------------------------
        # 11. Place agents in gridworld
        # --------------------------------------------------
        agent_id = 2
        for pos in self.start_pos:
            x, y = pos.tolist()
            self.gridworld[x, y] = agent_id
            agent_id += 1

        # --------------------------------------------------
        # 12. Place goals in goalworld
        # --------------------------------------------------
        agent_id = 2
        for pos in self.goal_pos:
            x, y = pos.tolist()
            self.goalworld[x, y] = agent_id
            agent_id += 1
    def _structured_world_2(self):
        """
        STRUCTURED EVALUATION WORLD

        - Boxes ONLY on TOP and BOTTOM
        - Boxes are perfectly tiled
        - NO gaps (including right edge)
        - Separator between boxes is EXACTLY 1 grid cell thick
        - Walls are 1 cell thick
        - Exactly ONE opening per box
        - MULTIPLE goals inside each box
        - Agents start freely on LEFT
        """

        # self.lock.acquire()
        try:
            self.finished.fill_(False)

            SIZE = int(self.SIZE.item()) if isinstance(self.SIZE, torch.Tensor) else int(self.SIZE)
            N    = int(self.num_agents.item()) if isinstance(self.num_agents, torch.Tensor) else int(self.num_agents)

            # --------------------------------------------------
            # Initialize empty world
            # --------------------------------------------------
            self.gridworld.zero_()
            self.obstacle_world.zero_()
            self.goalworld.zero_()

            # --------------------------------------------------
            # LEFT spawn area
            # --------------------------------------------------
            spawn_w = max(4, SIZE // 6)

            spawn_cells = torch.cartesian_prod(
                torch.arange(SIZE),
                torch.arange(spawn_w)
            )
            perm = torch.randperm(spawn_cells.size(0))

            self.start_pos.zero_()
            self.start_pos.copy_(spawn_cells[perm[:N]])

            # --------------------------------------------------
            # Box parameters
            # --------------------------------------------------
            wall = 1
            box_height = max(10, SIZE // 6)

            start_col = max(spawn_w + 2, SIZE // 4)
            end_col   = SIZE - 1

            min_box_width = max(8, 2 * wall + 3)

            available_width = end_col - start_col + 1
            n_boxes = max(2, available_width // min_box_width)

            base_box_width = available_width // n_boxes
            remainder      = available_width % n_boxes

            boxes = []

            # --------------------------------------------------
            # Build TOP & BOTTOM boxes (shared 1-cell separators)
            # --------------------------------------------------
            cur_c = start_col
            for i in range(n_boxes):
                bw = base_box_width + (1 if i < remainder else 0)

                c0 = cur_c
                c1 = cur_c + bw - 1

                # TOP box
                boxes.append({
                    "r0": 0,
                    "r1": box_height - 1,
                    "c0": c0,
                    "c1": c1,
                    "door": "bottom"
                })

                # BOTTOM box
                boxes.append({
                    "r0": SIZE - box_height,
                    "r1": SIZE - 1,
                    "c0": c0,
                    "c1": c1,
                    "door": "top"
                })

                # IMPORTANT: next box starts on SAME wall column
                cur_c = c1

            # --------------------------------------------------
            # Draw boxes and openings
            # --------------------------------------------------
            for box in boxes:
                r0, r1 = box["r0"], box["r1"]
                c0, c1 = box["c0"], box["c1"]

                # Walls
                self.gridworld[r0:r1 + 1, c0:c1 + 1] = 1

                # Interior
                self.gridworld[r0 + wall:r1 - wall + 1,
                            c0 + wall:c1 - wall + 1] = 0

                # Single opening (centered)
                dc = (c0 + c1) // 2
                if box["door"] == "bottom":
                    self.gridworld[r1, dc] = 0
                    self.gridworld[r1 - 1, dc] = 0
                else:  # top
                    self.gridworld[r0, dc] = 0
                    self.gridworld[r0 + 1, dc] = 0

            # --------------------------------------------------
            # Place MULTIPLE goals inside boxes (WITH replacement)
            # --------------------------------------------------
            goal_cells = []
            for box in boxes:
                for r in range(box["r0"] + wall, box["r1"] - wall + 1):
                    for c in range(box["c0"] + wall, box["c1"] - wall + 1):
                        goal_cells.append((r, c))

            idx = torch.randint(0, len(goal_cells), (N,))
            self.goal_pos.zero_()
            for i in range(N):
                r, c = goal_cells[int(idx[i].item())]
                self.goal_pos[i, 0] = int(r)
                self.goal_pos[i, 1] = int(c)

            # --------------------------------------------------
            # Obstacle world
            # --------------------------------------------------
            self.obstacle_world.copy_((self.gridworld == 1).int())

            # --------------------------------------------------
            # Place agents and goals by ID
            # --------------------------------------------------
            self.agents_past.zero_()
            self.agents_past.copy_(self.start_pos)

            agent_id = 2
            for i in range(N):
                sx = int(self.start_pos[i, 0].item())
                sy = int(self.start_pos[i, 1].item())
                gx = int(self.goal_pos[i, 0].item())
                gy = int(self.goal_pos[i, 1].item())

                self.gridworld[sx, sy] = agent_id
                self.goalworld[gx, gy] = agent_id
                agent_id += 1

        finally:
            # self.lock.release()
            pass
    def _structured_world_3(self):
        """
        Structured world with square rigid obstacle blocks.
        No random single-cell noise.
        """

        # --------------------------------------------------
        # Safe scalar extraction
        # --------------------------------------------------
        SIZE = int(self.SIZE.item()) if isinstance(self.SIZE, torch.Tensor) else int(self.SIZE)
        N    = int(self.num_agents.item()) if isinstance(self.num_agents, torch.Tensor) else int(self.num_agents)

        self.finished = False

        # --------------------------------------------------
        # PARAMETERS (tune freely)
        # --------------------------------------------------
        BLOCK_SIZE    = 4    # side length of square obstacle blocks
        GAP_SIZE      = 2    # free space between blocks
        BORDER_CLEAR  = 2    # free border for starts/goals

        # --------------------------------------------------
        # Clear worlds
        # --------------------------------------------------
        self.gridworld.zero_()
        self.obstacle_world.zero_()
        self.goalworld.zero_()

        # --------------------------------------------------
        # 1. Build square obstacle grid
        # --------------------------------------------------
        stride = BLOCK_SIZE + GAP_SIZE

        for r in range(BORDER_CLEAR, SIZE - BORDER_CLEAR, stride):
            for c in range(BORDER_CLEAR, SIZE - BORDER_CLEAR, stride):
                r_end = min(r + BLOCK_SIZE, SIZE - BORDER_CLEAR)
                c_end = min(c + BLOCK_SIZE, SIZE - BORDER_CLEAR)

                self.gridworld[r:r_end, c:c_end] = 1  # solid square block

        # --------------------------------------------------
        # 2. Define obstacle world
        # --------------------------------------------------
        self.obstacle_world.copy_((self.gridworld == 1).int())

        # --------------------------------------------------
        # 3. Collect FREE cells for starts/goals
        # --------------------------------------------------
        free_cells = torch.stack(torch.where(self.gridworld == 0), dim=1)

        if free_cells.size(0) < 2 * N:
            raise RuntimeError("Not enough free cells for agents and goals")

        perm = torch.randperm(free_cells.size(0))

        # --------------------------------------------------
        # 4. Assign starts and goals (no overlap)
        # --------------------------------------------------
        self.start_pos.zero_()
        self.goal_pos.zero_()

        self.start_pos.copy_(free_cells[perm[:N]])
        self.goal_pos.copy_(free_cells[perm[N:2*N]])

        self.start_pos = self.start_pos.long()
        self.goal_pos  = self.goal_pos.long()

        # --------------------------------------------------
        # 5. Goal world
        # --------------------------------------------------
        self.goalworld.copy_(self.gridworld)

        # --------------------------------------------------
        # 6. Agents past
        # --------------------------------------------------
        self.agents_past.zero_()
        self.agents_past.copy_(self.start_pos)

        # --------------------------------------------------
        # 7. Place agents in gridworld
        # --------------------------------------------------
        agent_id = 2
        for i in range(N):
            x, y = self.start_pos[i].tolist()
            self.gridworld[x, y] = agent_id
            agent_id += 1

        # --------------------------------------------------
        # 8. Place goals in goalworld
        # --------------------------------------------------
        agent_id = 2
        for i in range(N):
            x, y = self.goal_pos[i].tolist()
            self.goalworld[x, y] = agent_id
            agent_id += 1

    def _get_blocking_reward(self,agent_id):
        '''calculates how many robots the agent is preventing from reaching goal
        and returns the necessary penalty'''
        #accumulate visible robots
        other_robots=[]
        other_locations=[]
        inflation=0
        # Get agent's position
        position = self.get_pos(agent_id)
        half_obs_size = torch.div(self.observation_size, 2, rounding_mode='floor')
        # Calculate the top-left and bottom-right corner of the observation window
        top_left=(position[0]-half_obs_size,position[1]-half_obs_size)
    
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        for agent in range(2,self.num_agents+2):
            if agent==agent_id: continue
            cur_pos1=self.get_pos(agent)
            x, y = cur_pos1  # Unpack for boundary checks
            if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
                continue
            other_robots.append(agent)
            other_locations.append(cur_pos1)
        num_blocking=0
        obstcle_world=self.obstacle_world.cpu().numpy()
        for agent in other_robots:
            cur_pos2=self.get_pos(agent)
            goal_pos2=self.get_goal(agent)
            for loc in other_locations:
                if torch.equal(loc,cur_pos2):
                    other_locations.remove(loc)
                    break
            #before removing
            path_before=self._astar(obstcle_world,cur_pos2,goal_pos2,robots=other_locations+[self.get_pos(agent_id)])
            
            #after removing
            path_after=self._astar(obstcle_world,cur_pos2,goal_pos2,robots=other_locations)
            other_locations.append(self.get_pos(agent))
            if (path_before is None and path_after is None):continue
            if (path_before is not None and path_after is None):continue
            if (path_before is None and path_after is not None)\
                or len(path_before)>len(path_after)+inflation:
                num_blocking+=1
        
        if num_blocking==0:
            blocking_penalty=0.0
        else:
             blocking_penalty=  num_blocking*self.BLOCKING_COST  
        return blocking_penalty
    
    def _astar1(self,world,start,goal,robots):
        '''robots is a list of robots to add to the obstacle world'''
        world_copy = np.copy(world)  # Make a fresh copy
        for robot in robots:
            i, j = robot.tolist()  # Convert tensor position to list to index into the world
            world_copy[i, j] = 1  # Mark as obstacle
        try:
            start=start.cpu().numpy().tolist()
            goal=goal.cpu().numpy().tolist()
            print("here")
            path=cpp_mstar.find_path(world_copy,[start],[goal],1,5)
        except NoSolutionError:
            path=None
        for (i,j) in robots:
            world[i,j]=0
        return path
    
    def _astar2(self, world, start, goal, robots):
        """
        A pure Python A* implementation to replace cpp_mstar.find_path.
        Robots are treated as dynamic obstacles.
        """

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Mark robots as obstacles
        world_copy = np.copy(world)
        for robot in robots:
            i, j = robot.tolist()
            world_copy[i, j] = 1

        start = tuple(start.cpu().numpy().tolist())
        goal = tuple(goal.cpu().numpy().tolist())

        if world_copy[start[0]][start[1]] == 1 or world_copy[goal[0]][goal[1]] == 1:
            return None  # Start or goal is blocked

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))

        came_from = {}
        g_score = {start: 0}
        visited = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < world.shape[0] and 0 <= neighbor[1] < world.shape[1]:
                    if world_copy[neighbor[0]][neighbor[1]] == 1:
                        continue

                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                        came_from[neighbor] = current

        return None  # No path found
    
    def _astar(self, world, start, goal, robots):
        """
        A* pathfinding using the same action space as the environment (including diagonals).
        Robots are treated as dynamic obstacles.
        Returns a list of positions from start to goal.
        """
       

        def heuristic1(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Mark robots as obstacles
        world_copy = np.copy(world)
        for robot in robots:
            i, j = robot.tolist()
            world_copy[i, j] = 1

        start = tuple(start.cpu().numpy().tolist())
        goal = tuple(goal.cpu().numpy().tolist())

        if world_copy[start[0]][start[1]] == 1 or world_copy[goal[0]][goal[1]] == 1:
            return None  # Start or goal is blocked

        open_set = []
        heapq.heappush(open_set, (0 + heuristic1(start, goal), 0, start))

        came_from = {}
        g_score = {start: 0}
        visited = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path of positions
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < world.shape[0] and 0 <= neighbor[1] < world.shape[1]:
                    if world_copy[neighbor[0]][neighbor[1]] == 1:
                        continue

                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic1(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                        came_from[neighbor] = current

        return None  # No path found

    def _done(self):
        "To check all agents reach its goal"
        numComplete = 0
        for agent in range(2,self.num_agents+2):
            agent_pos = self.get_pos(agent)
            if self.goalworld[agent_pos[0],agent_pos[1]].item() == agent:
                numComplete += 1
                
        return numComplete==self.num_agents
    
 

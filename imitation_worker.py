import numpy as np
from collections import OrderedDict
from threading import Lock
import sys
import random
import math
import copy
from train import train
sys.path.append('../')
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import multiprocessing
from ACNet import A3CNetwork
import scipy.signal as signal
import gc
def ensure_shared_grads(local_model, shared_model,device):
     for local_param, shared_param in zip(local_model.parameters(), shared_model.parameters()):
            
            if local_param.grad is None:
                lparam = None
                continue
            else:
                lparam = local_param.grad if device == 'cpu' else local_param.grad.cpu()  # grad moved to cpu

            # Check if shared model's gradient exists. If not copy grads else accumulate it
            if shared_param.grad is None: 
                shared_param._grad = lparam
            else:
                shared_param._grad += lparam
                # return

def expert_worker(agent_id,master_network,optimizer,imit_loss_per_ep,device,RNN_SIZE,a_size, shared_env,num_agents,mstar_path,shared_results,stop_flag, lock, barrier):
     
    try:
            observation_size = 10
        
            local_AC = A3CNetwork(observation_size,a_size).to(device)
            with lock:
                local_AC.load_state_dict(master_network.state_dict())
                
            stop_flag.value = 0  # Set the stop flag to 0 to signal that all agents can start exploring
            barrier.wait()
            
            
            return_mstar_path = [None] * num_agents # to store path return by the mstar algorithm
                
                    
            Obstacle_world=shared_env.obstacle_world
            starts=shared_env.start_pos.tolist()
            
            
            ends=shared_env.goal_pos.tolist()
                    
            if agent_id==2: #Only one agent is used to get expert
                    try:
                    
                        return_mstar_path=cpp_mstar.find_path(Obstacle_world,starts,ends,2,5)
                        
                    except OutOfTimeError:
                        #M* timed out 
                        print("timeout")
                    except NoSolutionError:
                        print("no solution????")
                    except Exception as e:
                        print(e)
                        
                    
                    # Clear the existing list to free up space for a new path from the expert.
                    mstar_path[:] = []
                    # Add new elements to the list so other agents can access it
                    mstar_path.extend(return_mstar_path)
                    
            barrier.wait()
            if mstar_path is None or len(mstar_path) == 0 or any(value is None for value in mstar_path):
                    if agent_id==2:
                        print("No path returned by the expert",flush=True)
                    return
                            
            else:
                #-------------recreate the expert path in the actual world----------------#
                barrier.wait()
                if agent_id==2:
                    print("Expert returned path successfully..",flush=True)
                    print("Started recreating the expert path in the actual grid world....",flush=True)
                

                agent_expert_path = [step[agent_id-2] for step in mstar_path]
                
                
                # print("path",agent_id,agent_expert_path)
                
                barrier.wait()#synchronize before further processing
                # Clear the existing list to free up space for sharing the results
                # of actions while recreating the expert path in the grid world.
                shared_results[:] = [] #to verify there is no None value when recreated expert path
                results=[] #for individual agents
                
                for t in range(len(agent_expert_path[:-1])):
                    
                    if stop_flag.value:  # Check if any agent has signaled to stop
                            # print(f'Agent {agent_id} stopping early due to a stuck agent.')
                            break
                   
                    observation=shared_env._observe(agent_id)
                    all_agent_moved=False
                    steps=0
                    while not all_agent_moved:
                        if stop_flag.value:  # Check if any agent has signaled to stop
                            # print(f'Agent {agent_id} stopping early due to a stuck agent.')
                            break
                        steps+=1
                        o=observation
                        pos=agent_expert_path[t] #take each agents position at the same timestep
                        newPos=agent_expert_path[t+1]#guaranteed to be in bounds by loop guard
                        direction=(newPos[0]-pos[0],newPos[1]-pos[1])
                        a=shared_env.getAction(direction)
                        agents_blocked=False
                        on_goal_count=0
                        _,_,_,on_goal,agents_blocked,valid_action=shared_env._step((agent_id,a))
                        if agents_blocked:
                             print(agent_id,"blocking from imitation learning")
                        

                        if steps>num_agents**2:  #maximum steps allowed to take:
                            # if we have a very confusing situation where lots of agents move in a circle
                            # (difficult to parse and also (mostly) impossible to learn)
                            print(f'agent {agent_id} stuck in a loop............')
                            # return None
                            stop_flag.value = 1  # Set the stop flag to signal all agents to stop
                            shared_results.append(None)
                            break
                        if not valid_action:
                            #Give the agent another chance to move bcz previous action is not allowed
                            all_agent_moved=False
                            continue
                        all_agent_moved=True #  agent moved successfuly
                        
                        #shared_results list is used for tracking any 1 agents returns none value
                        shared_results.append([o[0].cpu().numpy().tolist(),o[1].cpu().numpy().tolist(),a]) #throw error if it is not list
                        results.append([o[0],o[1],a])
                    barrier.wait() # important other agents should wait to see any agent is stuck or not
                                    # if any agent is stuck we can not use it for training
                
                barrier.wait()#synchronize before further processing
                  
                if shared_results is None or len(shared_results) == 0 or any(value is None for value in shared_results):
                    #if the result contain none(means agent stuck) in it we can not use it further process
                    if agent_id==2:
                        print(f'Failed  to recreate the expert path in the actual world cant use the results for further processing.',flush=True)
                    
                else:
                    if agent_id==2:
                        print(f'Successfully recreated the expert path in the actual world for all agents.',flush=True)
                    # RNN_SIZE=256
                    # c_init = torch.zeros(1,1,RNN_SIZE).to(device)
                    # h_init = torch.zeros(1,1,RNN_SIZE).to(device)

                    # state_init = (c_init, h_init) 
                    
                    rnn_state0            = None
                    inputs_rollout = []
                    goal_pos_rollout = []
                    optimal_actions_rollout = []
                    
                    for item in results:
                        
                        inp=item[0] #dtype=torch.float32
                        
                        gol=item[1] #dtype=torch.float32)
                        
                        
                        optimal_actions=item[2]

                        inputs_rollout.append(inp)
                        goal_pos_rollout.append(gol)
                        optimal_actions_rollout.append(optimal_actions)
                    
                    
                    
                    
                    inputs=torch.stack(inputs_rollout, dim=0).to(device)
                    goal_pos=torch.stack(goal_pos_rollout, dim=0).to(device)
                    optimal_actions  = torch.tensor(optimal_actions_rollout).to(device)
                    
                    local_AC.train()
                    
                    policy, _, _ ,_, _,_,policy_imit=local_AC(inputs,goal_pos,rnn_state0,True)
                    
                    
                    loss_function=nn.CrossEntropyLoss()
                    imitation_loss = loss_function(policy_imit,optimal_actions)
                    imitation_loss.backward()

                    max_grad_norm = 1000.0
                    # parameters = list(local_AC.parameters())

                    # Compute the total gradient norm before clipping
                    # total_norm = torch.norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]), p=2)

                    # print(f"{agent_id},Total Gradient Norm before clipping: {total_norm.item()}")
                    
                    # Manually push the gradients to the global network
                   
                    # torch.nn.utils.clip_grad_norm_(local_AC.parameters(), max_norm=3.0)
                       
                    with lock:
                        optimizer.zero_grad()
                        master_network.zero_grad()
                        ensure_shared_grads(local_AC,master_network,device)
                        # Compute the total gradient norm before clipping
                        # Assuming local_AC is your model
                        parameters = list(master_network.parameters())
                        total_norm = torch.norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]), p=2)
                        torch.nn.utils.clip_grad_norm_(master_network.parameters(),max_norm=max_grad_norm)
                        print(f"{agent_id},Total Gradient Norm after clipping: {total_norm.item()}")
                        imit_loss_per_ep.value+=imitation_loss.item()
                        try:
                            # optimization step
                            optimizer.step() 
                            
                            
                            # print(f"Expert demonstration observed by agent {agent_id}.",flush=True)
                            local_AC.zero_grad()
 
                        except RuntimeError as e:
                            print(f"Error in optimizer step: {e}",flush=True)
                            print(f"agent: {agent_id} Error during Expert demonstration",flush=True)   

                    local_AC.zero_grad()
                    
                    
                    #sync the local and master weights
                    with lock:
                        with torch.no_grad(): 
                            local_AC.load_state_dict(master_network.state_dict())         
                
                barrier.wait()
            
            
            
                    
                    
    except Exception as e:
        print(f"Error in process {agent_id}: {e}",flush=True)
    
    finally:
        # Optional: Clear CUDA memory explicitly
        torch.cuda.empty_cache()
        print(f"Process {agent_id} finished execution.",flush=True)

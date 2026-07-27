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
import torch.nn.functional as F
import time
import os
import sys
import multiprocessing
from ACNet import A3CNetwork
from train import train
import scipy.signal as signal
import gc


def normal_worker(invalid_actions_per_ep,no_of_collision_per_ep,on_goal_loss_per_ep,blocking_loss_per_ep,valid_loss_per_ep,entropy_loss_per_ep,value_loss_per_ep,policy_loss_per_ep,steps_per_ep,rewards_per_ep,losses_per_ep,device,optimizer,master_network,shared_env,a_size,RNN_SIZE,agent_id,lock,barrier):
     
    try:
        
        episode_count=0
        invalid_action_count=0
        idle_action_count=0
        EXPERIENCE_BUFFER_SIZE =128 #try smaller batch for better memory retain for lstm
        observation_size = 10
        local_AC = A3CNetwork(observation_size,a_size).to(device)
        local_AC.load_state_dict(master_network.state_dict()) #sync the weight at the beginning 
    
        NUM_BUFFERS=1 #256 steps/EXPERIENCE_BUFFER_SIZE so 1 episode memory will retained
        gamma=0.95 
        action_list=[]
        maximum_allowed_steps=256
        i_buf = 0 #for bootstrapping
        episode_buffers = [[] for _ in range(NUM_BUFFERS)] #for storing experience that we used to train
        s1Values = [None for _ in range(NUM_BUFFERS)] #Bootstrapping the value for the last state in a episode
    
        elapsed_time = -999
        cumulative_episodic_loss = 0
        good_episode_count=0
        while(episode_count<1):
            with lock:
                    invalid_actions_per_ep.value=0
                    policy_loss_per_ep.value =0 #policy loss for all agents for current episode
                    value_loss_per_ep.value =0  #value loss for all agents for current episode
                    entropy_loss_per_ep.value =0 #entropy loss for all agents for current episode
                    valid_loss_per_ep.value =0 #valid action loss for all agents for current episode
                    blocking_loss_per_ep.value=0 #blocking loss for all agents for current episode
                    on_goal_loss_per_ep.value =0 #on goal loss for all agents for current episode
                    losses_per_ep.value =0
                    rewards_per_ep.value=0
                    no_of_collision_per_ep.value=0
            start_time = time.time()
            barrier.wait()
            if agent_id==2:
                shared_env._reset_world()
                
            barrier.wait()
            
            episode_buff=[] # To store the experience after each step for agents
            
            
            no_of_collisons=0
            action_1_count=0
            action_2_count=0
            action_3_count=0
            action_4_count=0
            predicted_action_0_count=0
            predicted_action_1_count=0
            predicted_action_2_count=0
            predicted_action_3_count=0
            predicted_action_4_count=0
            validActions        = shared_env._listNextValidActions(agent_id)
            current_observation = shared_env._observe(agent_id)
            blocking            = shared_env.blocking
            reward = 0
            total_reward = 0 # Accumulate reward for the episode
            
            on_goal = torch.equal(shared_env.get_pos(agent_id),shared_env.get_goal(agent_id))
         
            
            state_init = None
            
            rnn_state = state_init
            rnn_state0= rnn_state
            
            step_count  = 0
            
            roll_out_loss = 0
            combined_reward=0
            while(not shared_env.finished):
                
                inputs=current_observation[0]        
                inputs=inputs.unsqueeze(0).to(device)
                goal_pos=current_observation[1]
                goal_pos=goal_pos.unsqueeze(0).to(device)
                
                local_AC.eval() 
                with torch.no_grad():
                    #Take an action using probabilities from policy network output.
                    a_dist, value, rnn_state,pred_blocking,pred_on_goal,policy_sig,policy_logits=local_AC(inputs,goal_pos,rnn_state,True)
                
             

                masked_policy_logits = policy_logits.clone()
                
                #a_dist=action distribution
                all_valid_actions,agent_col_action=shared_env._listAll_ValidActions(agent_id)
               

                # Get the valid action distribution and normalize it
                # valid_dist = a_dist[0, validActions]
                # valid_dist /= torch.sum(valid_dist)  # Re-normalize to sum to 1

                
                # Find the action with the highest probability 
                predicted_action = torch.argmax(a_dist.flatten()).item()
                action_list.append(predicted_action)
                if predicted_action==1:
                    predicted_action_1_count+=1
                if predicted_action==2:
                    predicted_action_2_count+=1
                if predicted_action==3:
                    predicted_action_3_count+=1
                if predicted_action==4:
                    predicted_action_4_count+=1
                if predicted_action==0:
                    predicted_action_0_count+= 1

                if predicted_action not in all_valid_actions:
                    no_of_collisons+=1
                if predicted_action==agent_col_action and agent_col_action!=0:
                    with lock:
                        no_of_collision_per_ep.value+=1 #tracking collisions

                if predicted_action not in validActions:
                    # print("wrong",predicted_action,validActions)
                    invalid_action_count += 1
                    with lock:
                        invalid_actions_per_ep.value+=1
                else:
                    if predicted_action>0:
                        # print("correct",predicted_action,validActions)
                        pass
                      
                #Initialize train_valid with zeros and set valid actions
                valid_action_mask = torch.zeros(a_size,dtype=torch.float32)
                #used to Encourages the model to assign higher probabilities to valid actions
                valid_action_mask[validActions] = 1.0
                invalid_action_mask = valid_action_mask.unsqueeze(0)  
                masked_policy_logits[invalid_action_mask == 0] = -float('inf')  
              
                masked_policy = F.softmax(masked_policy_logits, dim=-1)
               
                a = torch.multinomial(masked_policy.cpu(), 1).item()
                        
                if a==1:
                    action_1_count+=1
                if a==2:
                    action_2_count+=1
                if a==3:
                    action_3_count+=1
                if a==4:
                    action_4_count+=1
                if a==0:
                    idle_action_count+= 1
                
                train_value = torch.tensor(1.0)

                barrier.wait() #synchronize 
                _,reward,_,on_goal,blocking,valid_step = shared_env._step((agent_id, a),episode=episode_count)
               
                on_goal_val = float(on_goal.item()) if isinstance(on_goal, torch.Tensor) else float(on_goal)
                blocking_val = float(blocking.item()) if isinstance(blocking, torch.Tensor) else float(blocking)
                
                
                barrier.wait() #wait till all agents take a step

                # Get common observation for all agents after all individual actions have been performed
                if on_goal==True:
                    validActions     = shared_env._listNextValidActions(agent_id,a,on_goal,episode_count)
                else:
                    validActions     = shared_env._listNextValidActions(agent_id,a,episode_count)

                next_observation = shared_env._observe(agent_id)
                Done            = shared_env.finished
                total_reward+=reward #rewards per step for single agent
                with lock:
                        rewards_per_ep.value+=reward #combined rewards for all agents per step
                
                barrier.wait() #increment the steps together
                episode_buff.append([current_observation[0],a,reward,valid_action_mask,value[0],on_goal_val,blocking_val,current_observation[1],train_value])
                current_observation=next_observation
                
                if Done.item() == True: 
                   
                    if agent_id==2:
                        print(f'{step_count} Goodbye World. We did it! agent_id {agent_id} world size:{shared_env.SIZE} prob: {shared_env.PROB}',flush=True)
                
                # If the episode hasn't ended, but the experience buffer is full, then we
                # make we train the model using that experience rollout.
                if (len(episode_buff) % EXPERIENCE_BUFFER_SIZE == 0 or Done.item()):
                    
                    # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                    # Update the "episode buffers" 
                    if len(episode_buff) >= EXPERIENCE_BUFFER_SIZE:
                        #If episode_buffer has at least EXPERIENCE_BUFFER_SIZE, 
                        #store the most recent experiences in episode_buffers[i_buf].
                        episode_buffers[i_buf] = episode_buff[-EXPERIENCE_BUFFER_SIZE:]
                    else:
                        #Otherwise, the entire episode_buffer is stored
                        episode_buffers[i_buf] = episode_buff[:]

                    #Handling Bootstrapping for the Value Function:
                    if Done.item():
                        #If the episode is done (Done == True), all agents reached goal
                        #the final value estimate for the last state 
                        #(s1Values[i_buf]) is set to 0 because there's no future state.
                        s1Values[i_buf] = torch.tensor(0.0)
                    else:
                        #Otherwise estimate the current state's value 
                        #using the actor-critic network self.local_AC, 
                        #which predicts based on the input state, goal position, and current RNN state.
                        #Use this result to bootstrap the last state's value (s1Values[i_buf]).
                        with torch.no_grad(): 
                            inputs=current_observation[0]
                            inputs=inputs.unsqueeze(0).to(device)
                            goal_pos=current_observation[1]
                            goal_pos=goal_pos.unsqueeze(0).to(device)
                            _,s1Values[i_buf],_,_,_,_,_=local_AC(inputs,goal_pos,rnn_state,True)
                            s1Values[i_buf]=s1Values[i_buf][0]

                    #Random Buffer Selection:
                    #Randomly selects an experience buffer for training 
                    #to avoid overfitting to the most recent buffer.
                    if len(episode_buffers[i_buf]) > 0 and NUM_BUFFERS>1:
                        i_rand = random.randint(0,i_buf)
                        if len(episode_buffers[i_rand])==0:
                            i_rand = i_buf
                    else:
                        i_rand = i_buf  # Use the current buffer if the selected one is empty
                    
                    
                    #train using the selected experience
                    # if agent_id==2 and step_count==255:
                    #     # print("used action",idle_action_count,action_1_count,action_2_count,action_3_count,action_4_count)
                    #     # print("predcited action",predicted_action_0_count,predicted_action_1_count,predicted_action_2_count,predicted_action_3_count,predicted_action_4_count)
                    #     print(f'invalid action count: {invalid_actions_per_ep.value}  current step:{step_count}',flush=True)
                    # print(f'action dist {a_dist}  current step:{step_count}',flush=True)
                        
                    # print(agent_id,"predcited action",predicted_action_0_count,predicted_action_1_count,predicted_action_2_count,predicted_action_3_count,predicted_action_4_count)
                    # if no_of_collisons>1 or no_of_collision_per_ep.value>1:
                    #     print(agent_id,"no_of_collisons",no_of_collisons)
                    #     print(agent_id,"no_of_agent_collisons",no_of_collision_per_ep.value)
                    #     print(agent_id,"predcited action",predicted_action_0_count,predicted_action_1_count,predicted_action_2_count,predicted_action_3_count,predicted_action_4_count)
                    pol_los, val_los, entro_los, valid_los, block_los, on_gol_los, total_los= train(local_AC,master_network,optimizer,device,lock,barrier,episode_buffers[i_rand],a_size,agent_id,gamma,Done.item(),s1Values[i_rand],rnn_state0,invalid_actions_per_ep.value,episode_count)
                    action_list=[]
                    with lock:
                        #for logging and tensorboard
                        policy_loss_per_ep.value +=pol_los #policy loss for all agents for current episode
                        value_loss_per_ep.value +=val_los  #value loss for all agents for current episode
                        entropy_loss_per_ep.value +=entro_los #entropy loss for all agents for current episode
                        valid_loss_per_ep.value +=valid_los #valid action loss for all agents for current episode
                        blocking_loss_per_ep.value+=block_los #blocking loss for all agents for current episode
                        on_goal_loss_per_ep.value +=on_gol_los #on goal loss for all agents for current episode
                        losses_per_ep.value +=total_los ##combined losses for all agents for current episode
                
                    i_buf = (i_buf + 1) % NUM_BUFFERS 
                    
                    rnn_state0             = rnn_state
                    
                    episode_buffers[i_buf] = []  # Reset buffer
                    # invalid_action_count = 0  
                
                if step_count >= maximum_allowed_steps or Done.item():
                    invalid_action_count=0
                    
                    idle_action_count=0
                    break
                
                barrier.wait()
                step_count = step_count+1
                if agent_id==2:
                    steps_per_ep.value=step_count # to track how many steps agents took for tensorboard
                        
                
            cumulative_episodic_loss += roll_out_loss/step_count if step_count>0 else roll_out_loss
            barrier.wait()             
            episode_count=episode_count+1
            end_time = time.time()
            elapsed_time = end_time - start_time
            if agent_id==2:
                print(f'invalid action count: {invalid_actions_per_ep.value}  current episode elapsed_time:{elapsed_time}',flush=True)
    
    except Exception as e:
        print(f"Error in process {agent_id}: {e}",flush=True)
    
    finally:
        print(f"Process {agent_id} finished execution.",flush=True)

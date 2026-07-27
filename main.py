import numpy as np
import os
from collections import OrderedDict
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
import sys
import multiprocessing
from ACNet import A3CNetwork
from environment import MAPFEnv 
from imitation_worker import expert_worker
from normal_worker import normal_worker

import gc
from torch.utils.tensorboard import SummaryWriter
from test import run_simulations,test_a3c
import logging
from test_visualization import visualize_gif
from test_environment import testMAPFEnv
# function to log metrics
def log_to_file(filename, episode, total_reward, average_reward, total_loss,value_loss):
    with open(filename, "a") as f:
        f.write(f"Episode {episode}, Reward: {total_reward:.2f}, "
                f"Average Reward: {average_reward:.2f}, Loss: {total_loss:.2f}, value_loss: {value_loss:.2f}\n")



if __name__=='__main__':

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)  # Prevents excessive CPU usage
    mp.set_start_method('spawn')
    if torch.cuda.is_available():
        print("CUDA is available!",flush=True)
    else:
        print("CUDA is not available.",flush=True)
    
    # ---Learning parameters--- #
    ALLOWED_WORLD_SIZES = [10,20,30,40,50,60,70] # Define the allowable world sizes 
    a_size= 5 #action size number of actions agents can take
    demonstarion_threshold=0.5 #Threshold for determining imitation mode
    RNN_SIZE = 256 #LSTM hidden state size
    LR_Q = 1e-5 #Learning rate

    #---Choose Training Mode---#
    hybrid_mode=True #agents trains on both RL and IL modes
    imitation_only_mode=False #agents trains only from Imitation learning
    RL_only_mode=False #agents trains only from Reinforcement learning
    # Ensure only one mode is True
    mode_flags = [hybrid_mode, imitation_only_mode, RL_only_mode]
    if mode_flags.count(True) != 1:
        raise ValueError("Exactly one of [hybrid_mode, imitation_only_mode, RL_only_mode] must be set to True.")
    
    #--- Store logs ---#
    log_filename = "training_logs.txt" #for logging to local files
    log_dir = 'training_logs'  # for tensorboard 
    logging.basicConfig(filename='test_results.txt', level=logging.INFO, format='%(message)s')
    writer = SummaryWriter(log_dir)

    # ---Model and optimizer names---#
    model_path = 'master_network.pt'
    optimizer_path='optimizer_state.pt'
    
    experiment = 0
    while(experiment < 40000):
        num_agents=random.randint(2,10) # Choose the number of agents randomly before each episode starts
        World_SIZE=random.choice(ALLOWED_WORLD_SIZES)  # Choose the World size randomly before each episode starts
        PROB= random.uniform(0.1, 0.3) # Choose the Obstacle probability randomly before each episode starts
        print(f'Started experiment run {experiment} with world size {World_SIZE}',flush=True)
        print(f"Number of agents set to: {num_agents} and obstacle prob {PROB}",flush=True)
        observation_size = 10
        #--- Testing after every 100 episodes ---#
        try:
            if experiment > 0 and experiment%100==0:
                num_agentss=10
                env=testMAPFEnv(num_agentss,PROB=PROB,SIZE=World_SIZE,DIAGONAL_MOVEMENT=False)
                test_a3c_network = test_a3c(model_path, observation_size,a_size)
                grid_size = env.SIZE
                obs_density = env.PROB

                print(f"Test run {experiment} with world size {grid_size}", flush=True)
                print(f"Number of agents set to: {num_agentss} and Obstacle prob {obs_density}", flush=True)
                
                logging.info(f"Started Test {experiment}: Number of agents: {num_agentss}, World size: {grid_size}, Obstacle probability: {obs_density:.2f}")
                finished, success, steps_count, collisions_count, rewards_count, paths, metrics = run_simulations(
                    num_agentss, env, test_a3c_network, experiment, grid_size, obs_density
                )
                #Optional visualization call
                # if experiment%500==0:
                #     visualize_gif(num_agentss, grid_size, obs_density, env.obstacle_world, paths, env.start_pos.tolist(), env.goal_pos.tolist(), experiment)
                experiment += 1
                continue
        except RuntimeError as e:
            print(f"Test failed at Test Number {experiment}: {e}")
            experiment += 1
            continue
        
        
        # Determine training mode
        imitation = False

        if imitation_only_mode:
            print("Imitation-only mode active.", flush=True)
            imitation = True
        elif RL_only_mode:
            print("RL-only mode active.", flush=True)
            imitation = False
        elif hybrid_mode:
            demonstarion_prob = random.uniform(0, 1)
            if demonstarion_prob < demonstarion_threshold:
                print("Hybrid mode: running imitation learning this episode.", flush=True)
                imitation = True
            else:
                print("Hybrid mode: running reinforcement learning this episode.", flush=True)
                imitation = False
        else:
            raise ValueError("No training mode is enabled. Please set one mode to True.")
       
        lock = mp.Lock()
        barrier = mp.Barrier(num_agents)
        env=MAPFEnv(lock,num_agents,PROB=PROB,SIZE=World_SIZE,DIAGONAL_MOVEMENT=False)
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        manager = mp.Manager()
        rewards_per_ep = mp.Value('d', 0)  # to store the total rewards per episode
        losses_per_ep = mp.Value('d', 0)   # To store the total losses per episode
        steps_per_ep = mp.Value('i', 0)    #to track how many step agent took in 1 episode
                                            #need it if an episode end prematurly
        policy_loss_per_ep = mp.Value('d', 0)   # To store the total losses per episode
        entropy_loss_per_ep = mp.Value('d', 0)   # To store the total losses per episode
        value_loss_per_ep = mp.Value('d', 0)   # To store the total losses per episode
        valid_loss_per_ep = mp.Value('d', 0)   # To store the total losses per episode
        blocking_loss_per_ep = mp.Value('d', 0)   # To store the total losses per episode
        on_goal_loss_per_ep = mp.Value('d', 0)   # To store the total losses per episode 
        imit_loss_per_ep = mp.Value('d', 0)
        invalid_actions_per_ep = mp.Value('i', 0)   # To store the total losses per episode 
        no_of_collision_per_ep = mp.Value('i', 0)   # To store the total collision per episode 
        mstar_path= manager.list([]) #for storing path returned by the expert
        shared_results = manager.list() #For sharing the results of actions while recreating the expert path in the grid world.
        stop_flag = manager.Value('i', 0)  # Shared flag to stop all agents when 1 or more agent stuck in loop while recreating the expert path in the grid world.                                    
        
        #experience replay
        NUM_BUFFERS=50
        shared_episode_buffers = manager.list([[] for _ in range(NUM_BUFFERS)])
        shared_s1_values = manager.list([None for _ in range(NUM_BUFFERS)])
        shared_ibuf = mp.Value('i', 0)
        master_network = A3CNetwork(observation_size,a_size)
        if os.path.exists(model_path):
            master_network.load_state_dict(torch.load(model_path))
        master_network.share_memory()
    
        optimizer =  SharedAdam(master_network.parameters(), lr=LR_Q)#should we also save optimizer?
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
            
        master_network.zero_grad()
        optimizer.zero_grad()


        if imitation==True:
            device='cpu' #imitation Training only works in CPU
            processes = [mp.Process(target=expert_worker, args=(i,master_network,optimizer,imit_loss_per_ep,device,RNN_SIZE,a_size, env,num_agents,mstar_path,shared_results,stop_flag, lock, barrier)) for i in range(2,num_agents+2)]
        else:
            #Normal RL Training works in both GPU and CPU
            processes = [mp.Process(target=normal_worker, args=(invalid_actions_per_ep,no_of_collision_per_ep,on_goal_loss_per_ep,blocking_loss_per_ep,valid_loss_per_ep,entropy_loss_per_ep,value_loss_per_ep,policy_loss_per_ep,steps_per_ep,rewards_per_ep,losses_per_ep,device,optimizer,master_network,env,a_size,RNN_SIZE,i,lock,barrier)) for i in range(2,num_agents+2)]
        
        finished_flag = np.zeros(num_agents)
        try:
            # Start all processes
            for p in processes:
                p.start()
            
            while finished_flag.sum() < num_agents:
                # Wait for all processes to complete
                time.sleep(1)
                for i,p in enumerate(processes):
                    if p.is_alive():
                        #print(f"Process {p.pid} is still alive... waiting for it to finish...")
                        continue;
                    else:
                        print(f"Process {p.pid} is not alive, terminating it... attempt number: {finished_flag[i]}")
                        p.terminate()
                        p.join()
                        torch.cuda.empty_cache()
                        finished_flag[i] += 1
                        if finished_flag[i] > num_agents:
                            raise Exception

        except KeyboardInterrupt:
            print("Process interrupted by user.",flush=True)
            # Terminate all processes on user interrupt
            for p in processes:
                p.terminate()
                torch.cuda.empty_cache()
                
        except Exception as main_e:
            print(f"Error in main process: {main_e}",flush=True)
            # Terminate processes if an error occurs in the main block
            for p in processes:
                p.terminate()
                torch.cuda.empty_cache()

        finally:
            # Ensure all processes are cleaned up
            for p in processes:
                p.terminate()
                torch.cuda.empty_cache()
                p.join(timeout=1)

            print("All processes have been terminated and joined.",flush=True)
        
        # Log to file at the end of each episode
        total_reward=rewards_per_ep.value
        invalid_steps=invalid_actions_per_ep.value/num_agents
        Average_reward=total_reward/num_agents
        average_collision=no_of_collision_per_ep.value
        average_collision=average_collision/num_agents
        if not imitation:
            #save training logs to local files
            log_to_file(log_filename, experiment, total_reward, Average_reward, losses_per_ep.value/num_agents,value_loss_per_ep.value/num_agents)
        if imitation:
            writer.add_scalar('loss/imitation', imit_loss_per_ep.value/num_agents, experiment)
        else:
            writer.add_scalar('loss/policy', policy_loss_per_ep.value/num_agents, experiment)
            writer.add_scalar('loss/value', value_loss_per_ep.value/num_agents, experiment)
            writer.add_scalar('loss/entropy', entropy_loss_per_ep.value/num_agents, experiment)
            writer.add_scalar('loss/valid_action', valid_loss_per_ep.value/num_agents, experiment)
            writer.add_scalar('loss/on_goal', on_goal_loss_per_ep.value/num_agents, experiment)
            writer.add_scalar('loss/blocking', blocking_loss_per_ep.value/num_agents, experiment)
            writer.add_scalar('loss/Total', losses_per_ep.value/num_agents, experiment)
            writer.add_scalar('Steps per episode/total steps', steps_per_ep.value, experiment)
            writer.add_scalar('Steps per episode/invalid steps', invalid_steps, experiment)
            writer.add_scalar('Steps per episode/collision steps', average_collision, experiment)
            writer.add_scalar('Reward/Average', Average_reward, experiment)
            writer.add_scalar('Reward/Total', total_reward, experiment)
        
        print("Saving master,optimizer and clearing local variables before next experiment...")
        torch.save(master_network.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        del lock, barrier, env, manager, master_network, optimizer,rewards_per_ep,no_of_collision_per_ep,losses_per_ep,steps_per_ep,on_goal_loss_per_ep,blocking_loss_per_ep,valid_loss_per_ep,entropy_loss_per_ep,value_loss_per_ep,policy_loss_per_ep,invalid_actions_per_ep,imit_loss_per_ep,mstar_path,shared_results,stop_flag

        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished experiment run {experiment}...',flush=True)
        experiment += 1

    print(f"Training completed. Logs saved to {log_filename}")
    writer.close()
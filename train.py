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
import psutil

def ensure_shared_grads(local_model, shared_model,device):
     for local_param, shared_param in zip(local_model.parameters(), shared_model.parameters()):
            
            if local_param.grad is None:
                lparam = None
                continue
            else:
                lparam = local_param.grad if device == 'cpu' else local_param.grad.cpu()  # grad moved to cpu

            # Check if shared model's gradient exists. If not copy grads else accumulate it
            if shared_param.grad is None  or shared_param.grad.abs().sum() == 0: 
                
                shared_param._grad = lparam.clone()
            else:
                # print("here")
                shared_param._grad += lparam
                




# Discounted return (Monte Carlo style, for value loss)
def discount(x, gamma):
    result = torch.zeros_like(x)
    running = 0
    for t in reversed(range(len(x))):
        running = x[t] + gamma * running
        result[t] = running
    return result

# Compute Generalized Advantage Estimation (GAE) fully in PyTorch
def compute_gae(rewards, values, gamma, lam,device):
    """
    Compute Generalized Advantage Estimation (GAE) using only PyTorch tensor operations.
    """
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]  # TD error
    adv = torch.zeros_like(deltas, device=device)
    
    # Compute advantages via reversed cumulative sum
    for t in reversed(range(len(deltas))):
        adv[t] = deltas[t] + gamma * lam * adv[t + 1] if t + 1 < len(deltas) else deltas[t]
    
    return adv, values[:-1] + adv  # (advantages, target value)

def train(local_AC,master_network,optimizer,device,lock,barrier,rollout,a_size,agent_id, gamma,done, bootstrap_value, rnn_state0,invalid_actions_per_ep, ep, imitation=False):

    print(f'Hello from the local training worker: {agent_id}',flush=True)
    try:
        with torch.no_grad():
            # Convert rollout into tensors
            inputs_rollout = torch.stack([item[0] for item in rollout]).to(device)
            goal_pos_rollout = torch.stack([item[7] for item in rollout]).to(device)
            actions_rollout = torch.tensor([item[1] for item in rollout],  device=device)
            rewards_rollout = torch.tensor([item[2] for item in rollout],  device=device)
            valid_action_rollouts = torch.stack([item[3] for item in rollout]).to(device)
            values_rollout = torch.tensor([item[4].item() for item in rollout], device=device)
            train_val_rollouts = torch.cat([item[-1].unsqueeze(0) for item in rollout]).to(device)
            on_goals_rollouts = torch.tensor([item[5] for item in rollout], dtype=torch.float32, device=device)
            blockings_rollouts = torch.tensor([item[6] for item in rollout], dtype=torch.float32, device=device)
           
            # Extend values and rewards with the bootstrap value
            values_plus = torch.cat([values_rollout, torch.tensor([bootstrap_value.item()], device=device)])

            #-----# original tf implemnetation-------#---#
            # rewards_plus = torch.cat([rewards_rollout, torch.tensor([bootstrap_value.item()],device=device)])
            # discounted_returns = discount(rewards_plus, gamma=0.95)[:-1]  # drop final bootstrap value
            #----# modern Generalized Advantage Estimation (GAE) #------#
            gamma = 0.95 #Discount Factor
            lamda = 0.90 # GAE Smoothing Factor
            deltas = rewards_rollout + gamma * values_plus[1:] - values_plus[:-1] # The Temporal Difference (TD) Error#reward rollout not plus
            advantages = torch.zeros_like(deltas)
            gae = 0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + gamma * lamda * gae
                advantages[t] = gae

            # Final target for value loss
            target_value = values_rollout + advantages
            length=len(rollout)
            

            # One-hot encode actions
            actions_onehot = torch.nn.functional.one_hot(actions_rollout, num_classes=a_size).to(device)
    except RuntimeError as e:
        print(f"{agent_id} [ALLOC ERROR] during rollout preprocess: {e}")
        gc.collect()
        pl_l = val_l = entro_l = valid_l = block_l = on_gol_l = total_l = 0.0
        return pl_l, val_l, entro_l, valid_l, block_l, on_gol_l, total_l  
   

    # Forward pass through neural network
    local_AC.zero_grad()
    for p in local_AC.parameters():
        p.grad = None
    try:
        if rnn_state0 is not None:
            rnn_state0 = (rnn_state0[0].detach(), rnn_state0[1].detach())

        local_AC.train()
        policy, value_logits, state_out, blocking, on_goal, policy_sig, policy_logits = local_AC(
            inputs_rollout, goal_pos_rollout, rnn_state0, True)
        
        state_out = (state_out[0].detach(), state_out[1].detach())

    except RuntimeError as e:
        print(f"{agent_id} [ALLOC ERROR] during forward pass: {e}")
        # ✅ Clean up large tensors to free memory
        del inputs_rollout, goal_pos_rollout, actions_rollout, rewards_rollout,rnn_state0,values_plus,deltas,actions_onehot
        del valid_action_rollouts, values_rollout, train_val_rollouts,rollout,bootstrap_value
        del on_goals_rollouts, blockings_rollouts, advantages, target_value
        del local_AC
        gc.collect()
        pl_l = val_l = entro_l = valid_l = block_l = on_gol_l = total_l = 0.0
        return pl_l, val_l, entro_l, valid_l, block_l, on_gol_l, total_l    
    # Compute policy probabilities
    # Compute log probabilities using log_softmax for numerical stability
    log_probs_policy_logits = F.log_softmax(policy_logits, dim=-1)
    responsible_outputs = torch.sum(log_probs_policy_logits * actions_onehot, dim=1)
    # print(f"{agent_id} before clipping responsible_outputs : {responsible_outputs}")
    
    
    responsible_outputs = torch.clamp(responsible_outputs, min=-5.0)
    max_grad_norm = 1000.0
    advantages = torch.clamp(advantages, -5, 5)  # Prevents extreme policy updates
    # Loss computations
    BCE_loss = nn.BCEWithLogitsLoss(reduction='sum')
    # Compute entropy to encourage exploration
 
    entropy = -torch.sum(policy * log_probs_policy_logits) # Standard entropy formula
   
    policy_loss = -torch.sum(advantages * responsible_outputs)
    value_loss = ((target_value - value_logits.view(-1)) ** 2).sum()
    valid_action_loss = BCE_loss(policy_logits, valid_action_rollouts)
    blocking_loss = BCE_loss(blocking.squeeze(-1), blockings_rollouts)
    on_goal_loss = BCE_loss(on_goal.squeeze(-1), on_goals_rollouts) 
    
    # Compute total loss
    total_loss = (policy_loss + 0.5 * value_loss + 0.1*valid_action_loss+0.5 * blocking_loss + 0.5 * on_goal_loss - 0.05 * entropy).to(device)
    
    # Backpropagation
    try:
        total_loss.backward()
    except RuntimeError as e:
        print(f"{agent_id} [ALLOC ERROR] during back prop: {e}")
        # ✅ Clean up large tensors to free memory
        del inputs_rollout, goal_pos_rollout, actions_rollout, rewards_rollout,rnn_state0,values_plus
        del valid_action_rollouts, values_rollout, train_val_rollouts,rollout,deltas,actions_onehot
        del on_goals_rollouts, blockings_rollouts, advantages, target_value,bootstrap_value
        del policy, value_logits,state_out, policy_logits, blocking, on_goal, policy_sig, responsible_outputs
        del policy_loss,value_loss,entropy,valid_action_loss,blocking_loss,on_goal_loss,total_loss
        del local_AC
        gc.collect()
        pl_l = val_l = entro_l = valid_l = block_l = on_gol_l = total_l = 0.0
        return pl_l, val_l, entro_l, valid_l, block_l, on_gol_l, total_l
    parameters = list(local_AC.parameters())
            
    total_norm = torch.norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]), p=2)
         
    try:
    
        with lock:
            master_network.zero_grad()  # Reset gradients
            optimizer.zero_grad()
            
            
            ensure_shared_grads(local_AC,master_network,device)
            
            parameters = list(master_network.parameters())
            
            total_norm = torch.norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]), p=2)
            
            # print(f"{agent_id},Total Gradient Norm before clipping: {total_norm.item()}")
            
            
        
            parameters = list(master_network.parameters())
            
            total_norm = torch.norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]), p=2)
            
            # print(f"{agent_id},Total Gradient Norm after clipping: {total_norm.item()}")  
            torch.nn.utils.clip_grad_norm_(master_network.parameters(),max_norm=max_grad_norm) 
            optimizer.step() 
          
        
       
           
            local_AC.zero_grad() # Reset gradients
            for p in local_AC.parameters():
                p.grad = None
            print("Agent :",agent_id,"trained",flush=True)
            
    except RuntimeError as e:
        print(f"{agent_id} [ALLOC ERROR] during optimizer step: {e}")
        local_AC.zero_grad() # Reset gradients
        for p in local_AC.parameters():
            p.grad = None
        # ✅ Clean up large tensors to free memory
        del inputs_rollout, goal_pos_rollout, actions_rollout, rewards_rollout,rnn_state0,values_plus
        del valid_action_rollouts,state_out, values_rollout, train_val_rollouts,rollout,deltas,actions_onehot
        del on_goals_rollouts, blockings_rollouts, advantages, target_value,bootstrap_value
        del policy, value_logits, policy_logits, blocking, on_goal, policy_sig, responsible_outputs
        del policy_loss,value_loss,entropy,valid_action_loss,blocking_loss,on_goal_loss,total_loss
        del local_AC
        gc.collect()
        
        pl_l = val_l = entro_l = valid_l = block_l = on_gol_l = total_l = 0.0
        return pl_l, val_l, entro_l, valid_l, block_l, on_gol_l, total_l
        


    
    #sync the local and master weights
    with lock:
        with torch.no_grad(): 
            local_AC.load_state_dict(master_network.state_dict())

    
    
   
    
        
    
    #for tensorboard
    if length > 0:
        pl_l = policy_loss.item()
        val_l = value_loss.item()
        entro_l = entropy.item()
        valid_l = valid_action_loss.item()
        block_l = blocking_loss.item()
        on_gol_l = on_goal_loss.item()
        total_l = total_loss.item()
        
    else:
        # Return zero when length is zero
        pl_l = val_l = entro_l = valid_l = block_l = on_gol_l = total_l = 0.0
    
    

     # ✅ Clean up large tensors to free memory
    del inputs_rollout, goal_pos_rollout, actions_rollout, rewards_rollout,rnn_state0,values_plus
    del valid_action_rollouts, values_rollout, train_val_rollouts,rollout,deltas,actions_onehot
    del on_goals_rollouts, blockings_rollouts, advantages, target_value,bootstrap_value
    del policy, value_logits,state_out, policy_logits, blocking, on_goal, policy_sig, responsible_outputs
    del policy_loss,value_loss,entropy,valid_action_loss,blocking_loss,on_goal_loss,total_loss
    del local_AC
    gc.collect()


    return pl_l, val_l, entro_l, valid_l, block_l, on_gol_l, total_l
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import os
import glob
import time
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from ACNet import A3CNetwork


# Configure logging
logging.basicConfig(filename='test_results.txt', level=logging.INFO, format='%(message)s')

class test_a3c():
    def __init__(self, model_path, grid_size,a_size):
        self.grid_size = grid_size
        observation_size = 10
        self.network = A3CNetwork(observation_size, a_size)
        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()

    def set_env(self, env):
        self.env = env
        self.num_agents = self.env.num_agents
        self.agent_states = None
        self.size = self.env.SIZE
        self.init_world = self.env.obstacle_world
        self.path = []
        self.start_positions = env.start_pos
        self.goals = env.goal_pos
        self.path.append(self.start_positions.tolist())

    def step_all_parallel(self):
        inputs = []
        goal_pos = []
        for agent in range(2, self.num_agents + 2):
            o = self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])

        inputs = torch.stack(inputs)
        goal_pos = torch.stack(goal_pos)

        rnn_state0 = self.agent_states
        self.network.eval()
        with torch.no_grad():
            policy_vec, _, state_out, _, _, _, policy_logits = self.network(inputs, goal_pos, rnn_state0, training=False)

        self.agent_states = state_out

        temp_path = []
        no_of_collisons = 0
        total_reward = 0
        total_on_goal = 0
        total_blocking = 0
        robot_collisions = 0
        obstacle_collisions = 0

        for agent in range(2, self.num_agents + 2):
            # masked_policy_logits = policy_logits[agent - 2].clone().squeeze()
            # validActions = self.env._listNextValidActions(agent)
            # for i in range(a_size):
            #     if i not in validActions:
            #         masked_policy_logits[i] = -float('inf')
            # logits = policy_logits[agent - 2]  # shape: [a_size]
            # Create additive mask: 0 for valid, -inf for invalid
            # mask = torch.full_like(logits, float('-inf'))
            # mask[validActions] = 0
            # masked_policy_logits = logits + mask
            # masked_a_dist = torch.softmax(masked_policy_logits, dim=-1)
            action = torch.argmax(policy_vec[agent - 2]).item()
            # action = torch.argmax(masked_a_dist).item()
            _, reward, _, on_goal, blocking, valid_action = self.env._step((agent, action))

            total_reward += reward
            if on_goal:
                total_on_goal += 1
            if blocking:
                total_blocking += 1
            if not valid_action:
                no_of_collisons += 1
            pos = self.env.get_pos(agent)
            temp_path.append(pos.tolist())

        self.path.append(temp_path)
        return no_of_collisons, total_reward, total_on_goal, total_blocking

    def find_path(self, max_step=256):
        step = 0
        no_of_collisons = 0
        total_rewards = 0
        total_on_goal = 0
        total_blocking = 0
        while not self.env._done() and step < max_step:
            collisions, rewards, on_goal, blocking = self.step_all_parallel()
            no_of_collisons += collisions
            total_rewards += rewards
            total_on_goal += on_goal
            total_blocking += blocking
            step += 1
        if self.env._done():
            print("Goodbye World We Did It")
        return self.env._done(), step, no_of_collisons, total_rewards, self.path, total_on_goal, total_blocking


def run_simulations(num_agents, world, test_a3c_network, Total_Experiment, grid_size, obs_density):
    test_a3c_network.set_env(world)
    start_time = time.time()
    results = None #dict()
    success = 0
    finished = False

    try:
        finished, steps_took, collisions, total_rewards, paths, on_goal_count, blocking_count = test_a3c_network.find_path()
        success = int(finished)
    except (OutOfTimeError, NoSolutionError):
        steps_took = 0
        collisions = 0
        total_rewards = 0
        paths = []
        on_goal_count = 0
        blocking_count = 0
        finished = False

    time_taken = time.time() - start_time
    collisions=collisions/num_agents
    total_rewards=total_rewards/num_agents
    on_goal_count=on_goal_count/num_agents
    blocking_count=blocking_count/num_agents

 

    # Log per-episode metrics
    logging.info(
        f"[EPISODE {Total_Experiment}] Success: {finished} | Steps: {steps_took} || Collisions: {collisions} | "
        f"| Reward: {total_rewards} | "
        f"On Goal: {on_goal_count} | Blocking: {blocking_count} | Time: {time_taken:.2f}s"
    )

    return finished, success, steps_took, collisions, total_rewards, paths, results




        
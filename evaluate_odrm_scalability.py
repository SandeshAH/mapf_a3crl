import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import os
import glob
import time

from evaluation_environment import MAPFEnv
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from ACNet import A3CNetwork
from evaluation_visualization import visualize_gif

# Configure logging
logging.basicConfig(filename='odrm_evaluation_results.txt', level=logging.INFO, format='%(message)s')

class Evaluate_a3c():
    def __init__(self, model_path,action_filter, grid_size):
        self.grid_size = grid_size
        self.action_filter=action_filter
        observation_size = 10
        a_size = 5
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
            if self.action_filter:#filtered action mode or raw action mode
                validActions = self.env._listNextValidActions(agent)
                logits = policy_logits[agent - 2]  # shape: [a_size]
                # Create additive mask: 0 for valid, -inf for invalid
                mask = torch.full_like(logits, float('-inf'))
                mask[validActions] = 0
                masked_policy_logits = logits + mask
                masked_a_dist = torch.softmax(masked_policy_logits, dim=-1)
                action = torch.argmax(masked_a_dist).item()
                # print("filtered action execution mode")
            else:
                action = torch.argmax(policy_vec[agent - 2]).item()
                # print("Raw action execution mode")
            
            _, reward, _, on_goal, blocking, valid_action, robot_collision, obstacle_collision = self.env._step((agent, action))

            total_reward += reward
            if on_goal:
                total_on_goal += 1
            if blocking:
                total_blocking += 1
            if not valid_action:
                no_of_collisons += 1
            if robot_collision:
                robot_collisions += 1
            if obstacle_collision:
                obstacle_collisions += 1

            pos = self.env.get_pos(agent)
            temp_path.append(pos.tolist())

        self.path.append(temp_path)
        return no_of_collisons, total_reward, total_on_goal, total_blocking, robot_collisions, obstacle_collisions

    def find_path(self, max_step=256):
        step = 0
        no_of_collisons = 0
        total_rewards = 0
        total_on_goal = 0
        total_blocking = 0
        total_robot_collisions = 0
        total_obstacle_collisions = 0

        while not self.env._done() and step < max_step:
            collisions, rewards, on_goal, blocking, robot_col, obstacle_col = self.step_all_parallel()
            print(f'No of steps took so far: {step}/{max_step}')
            no_of_collisons += collisions
            total_rewards += rewards
            total_on_goal += on_goal/self.num_agents
            total_blocking += blocking/self.num_agents
            total_robot_collisions += robot_col
            total_obstacle_collisions += obstacle_col
            step += 1
        if self.env._done():
            print("Goodbye World We Did It")
        return self.env._done(), step, no_of_collisons, total_rewards, self.path, total_on_goal, total_blocking, total_robot_collisions, total_obstacle_collisions


def run_simulations(num_agents, world, a3c_model, Total_Experiment, grid_size, obs_density):
    a3c_model.set_env(world)
    start_time = time.time()
    results = dict()
    success = 0
    finished = False
    min_steps=256
    max_steps=4096
    max_step =2500 #max(min_steps, min(max_steps, 10 * num_agents))

    try:
        finished, steps_took, collisions, total_rewards, paths, on_goal_count, blocking_count, robot_col_total, obstacle_col_total = a3c_model.find_path(max_step=max_step)
        success = int(finished)
    except (OutOfTimeError, NoSolutionError):
        steps_took = 0
        collisions = 0
        total_rewards = 0
        paths = []
        on_goal_count = 0
        blocking_count = 0
        robot_col_total = 0
        obstacle_col_total = 0
        finished = False

    if not finished:
        num_complete = 0
        for agent in range(2,world.num_agents+2):
            agent_pos = world.get_pos(agent)
            if world.goalworld[agent_pos[0],agent_pos[1]].item() == agent:
                num_complete += 1
        
    else:
        num_complete=world.num_agents

    time_taken = time.time() - start_time
    collisions=collisions/num_agents
    total_rewards=total_rewards/num_agents
    on_goal_count=on_goal_count/num_agents
    blocking_count=blocking_count/num_agents
    robot_col_total=robot_col_total/num_agents
    obstacle_col_total=obstacle_col_total/num_agents
    results.update({
        'finished': finished,
        'agents_completed': num_complete,
        'agents_total': world.num_agents,
        'steps': steps_took,
        'collisions': collisions,
        'reward': total_rewards,
        'on_goal': on_goal_count,
        'blocking': blocking_count,
        'robot_collisions': robot_col_total,
        'obstacle_collisions': obstacle_col_total,
        'time': time_taken
    })

    # Log per-episode metrics
    logging.info(
        f"[EPISODE {Total_Experiment}] Success: {finished} | Steps: {steps_took} || Collisions: {collisions} | Agents Completed: {num_complete}/{world.num_agents} |"
        f"Robot: {robot_col_total} | Obstacle: {obstacle_col_total} || Reward: {total_rewards} | "
        f"On Goal: {on_goal_count} | Blocking: {blocking_count} | Time: {time_taken:.2f}s"
    )

    return finished, success, steps_took, collisions, total_rewards, paths, results,num_complete



if __name__ == "__main__":
   
    model_path = "models/hybrid_standard/master_network.pt"

    Total_Experiment = 1
    Successful_experiment = 0

    env_dir = "structured_world1"
    for env_path in glob.glob(f"{env_dir}/*.pkl"):

        # ---------- LOAD ENV ---------- #
        name = os.path.splitext(os.path.basename(env_path))[0]
        world = MAPFEnv()
        world.import_env(env_path)

        num_agents = world.num_agents
        grid_size = world.SIZE
        obs_density = world.PROB

        print(f"Started experiment run {Total_Experiment} with world size {grid_size}", flush=True)
        print(f"Number of agents set to: {num_agents} and obstacle prob {obs_density}", flush=True)

        logging.info(
            f"[Started experiment run {Total_Experiment}] "
            f"Agents: {num_agents} | Grid: {grid_size} | ObsProb: {obs_density:.2f}"
        )

        # ---------- PREPARE INPUT ---------- #
        Obstacle_world = world.obstacle_world
        starts = world.start_pos.tolist()
        goals = world.goal_pos.tolist()

        return_mstar_path = None
        failure_reason = None

        # ---------- RUN EXPERT ---------- #
        start_time = time.perf_counter()

        try:
            return_mstar_path = cpp_mstar.find_path(
                Obstacle_world,
                starts,
                goals,
                2,   # recursive (truthy → True)
                60    # time in sec
            )

        except OutOfTimeError:
            failure_reason = "timeout"

        except NoSolutionError:
            failure_reason = "no_solution"

        except Exception as e:
            # catches std::bad_alloc and any C++ crash
            failure_reason = str(e)

        runtime = time.perf_counter() - start_time

        # ---------- EVALUATE RESULT ---------- #
        success = (
            return_mstar_path is not None
            and isinstance(return_mstar_path, (list, tuple))
            and len(return_mstar_path) > 0
            and not any(v is None for v in return_mstar_path)
            and failure_reason is None
        )

        if not success:
            print("No path returned by the expert", flush=True)

            logging.info(
                f"[EPISODE {Total_Experiment}] "
                f"Success: False | "
                f"Steps: None | "
                f"Runtime_sec: {runtime:.4f} | "
                f"Failure_reason: {failure_reason}"
            )

        else:
            steps = len(return_mstar_path)
            Successful_experiment += 1

            print("Expert returned path successfully.", flush=True)
            print(f"Number of steps taken by expert: {steps}", flush=True)

            logging.info(
                f"[EPISODE {Total_Experiment}] "
                f"Success: True | "
                f"Steps: {steps} | "
                f"Runtime_sec: {runtime:.4f}"
            )

        Total_Experiment += 1

    # ---------- FINAL SUMMARY ---------- #
    print("Finished all tests!", flush=True)
    success_rate = (Successful_experiment / (Total_Experiment - 1)) * 100 if Successful_experiment > 0 else 0
    logging.info(
        f"\nALL EXPERIMENTS DONE\n"
        f"Success_rate: {success_rate}%"
    )

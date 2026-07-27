import tensorflow as tf
from ACNet import ACNet
import numpy as np
import json
import os
import mapf_gym_cap as mapf_gym
import time
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import shutil
#import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import re
import logging
# Configure logging
logging.basicConfig(filename='scalability_test_size70x70_agents_50_obs_30_envs_primalTF.txt', level=logging.INFO, format='%(message)s')


def load_env_from_pytorch_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    grid = data['gridworld'].numpy() if hasattr(data['gridworld'], 'numpy') else data['gridworld']
    start = data['start_pos'].numpy() if hasattr(data['start_pos'], 'numpy') else data['start_pos']
    goal = data['goal_pos'].numpy() if hasattr(data['goal_pos'], 'numpy') else data['goal_pos']
    # Remap obstacle: 1 -> -1 (for TF)
    grid = np.where(grid == 1, -1, grid)
    
    # Clear agent and goal spots
    for pos in start:
        grid[pos[0], pos[1]] = 0
    for pos in goal:
        grid[pos[0], pos[1]] = 0

    # Remap agents from 2+ -> 1+ for TF
    world = grid.copy()
    goals = np.zeros_like(world)

    for i, pos in enumerate(start):
        world[pos[0], pos[1]] = i + 1  # agent IDs start at 1

    for i, pos in enumerate(goal):
        goals[pos[0], pos[1]] = i + 1

    print("✅ Converted PyTorch pickle to TF-compatible world and goals.")
    return world, goals
class PRIMAL(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self,model_path,grid_size):
        self.grid_size=grid_size
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        self.sess=tf.Session(config=config)
        self.network=ACNet("global",5,None,False,grid_size,"global")
        #load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess,ckpt.model_checkpoint_path)
        
    def set_env(self,gym):
        self.num_agents=gym.num_agents
        self.agent_states=[]
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)
        self.size=gym.SIZE
        self.env=gym
        
    def step_all_parallel(self):
        action_probs=[None for i in range(self.num_agents)]
        '''advances the state of the environment by a single step across all agents'''
        #parallel inference
        actions=[]
        inputs=[]
        goal_pos=[]
        for agent in range(1,self.num_agents+1):
            o=self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])
        #compute up to LSTM in parallel
        h3_vec = self.sess.run([self.network.h3], 
                                         feed_dict={self.network.inputs:inputs,
                                                    self.network.goal_pos:goal_pos})
        h3_vec=h3_vec[0]
        rnn_out=[]
        #now go all the way past the lstm sequentially feeding the rnn_state
        for a in range(0,self.num_agents):
            rnn_state=self.agent_states[a]
            lstm_output,state = self.sess.run([self.network.rnn_out,self.network.state_out], 
                                         feed_dict={self.network.inputs:[inputs[a]],
                                                    self.network.h3:[h3_vec[a]],
                                                    self.network.state_in[0]:rnn_state[0],
                                                    self.network.state_in[1]:rnn_state[1]})
            rnn_out.append(lstm_output[0])
            self.agent_states[a]=state
        #now finish in parallel
        policy_vec=self.sess.run([self.network.policy], 
                                         feed_dict={self.network.rnn_out:rnn_out})
        policy_vec=policy_vec[0]
        no_of_collisions = 0
        total_reward = 0
        total_on_goal = 0
        total_blocking = 0
        robot_collisions = 0
        obstacle_collisions = 0
        for agent in range(1,self.num_agents+1):
            action=np.argmax(policy_vec[agent-1])
            _, reward, done, _, on_goal, blocking, valid_action,robot_collision, obstacle_collision=self.env._step((agent,action))
            total_reward +=reward
            if on_goal:
                total_on_goal += 1
            if blocking:
                total_blocking += 1
            if not valid_action:
                no_of_collisions += 1
            if robot_collision:
                robot_collisions += 1
            if obstacle_collision:
                obstacle_collisions += 1
        return no_of_collisions,total_reward,total_on_goal,total_blocking,robot_collisions, obstacle_collisions
    
    
    
    def find_path(self, max_step=256, gif_name="primal_rollout.gif"):
        '''Run a full environment to completion, or until max_step steps.'''
        solution = []
        step = 0
        save_frames = False
        output_dir = "frames"
        no_of_collisons = 0
        total_rewards = 0
        total_on_goal = 0
        total_blocking = 0
        total_robot_collisions = 0
        total_obstacle_collisions = 0
        # Clean and create frames directory
        if save_frames:
            shutil.rmtree(output_dir, ignore_errors=True)
            os.makedirs(output_dir, exist_ok=True)

        # Run simulation
        while not self.env._complete() and step < max_step:
            timestep = []
            for agent in range(1, self.env.num_agents + 1):
                timestep.append(self.env.world.getPos(agent))
            solution.append(np.array(timestep))

            if save_frames:
                frame = render_env_grid(
                    self.env.world.state,
                    self.env.world.goals,
                    [self.env.world.getPos(i) for i in range(1, self.env.num_agents + 1)],
                    step
                )
                imageio.imwrite("{}/frame_{:04d}.png".format(output_dir, step), frame)

            collisions,rewards,on_goals,blockings, robot_col, obstacle_col=self.step_all_parallel()
            no_of_collisons += collisions
            total_rewards += rewards
            total_on_goal += on_goals
            total_blocking += blockings
            total_robot_collisions += robot_col
            total_obstacle_collisions += obstacle_col
            step += 1

        # Always capture the final step (complete or timeout)
        timestep = []
        for agent in range(1, self.env.num_agents + 1):
            timestep.append(self.env.world.getPos(agent))
        solution.append(np.array(timestep))

        if save_frames:
            print(step, "[Frames] Final frame")
            frame = render_env_grid(
                self.env.world.state,
                self.env.world.goals,
                [self.env.world.getPos(i) for i in range(1, self.env.num_agents + 1)],
                step
            )
            imageio.imwrite("{}/frame_{:04d}.png".format(output_dir, step), frame)

        # ✅ Create GIF
        if save_frames:
            # Ensure GIF directory exists
            gif_dir = os.path.dirname(gif_name)
            if gif_dir and not os.path.exists(gif_dir):
                os.makedirs(gif_dir)

            print("[GIF] Generating {} from {}/...".format(gif_name, output_dir))
            frame_files = sorted([
                os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")
            ])
            with imageio.get_writer(gif_name, mode='I', fps=5) as writer:
                for f in frame_files:
                    image = imageio.imread(f)
                    writer.append_data(image)
                # Freeze last frame longer
                for _ in range(5):  # Hold final frame 5 extra frames
                    writer.append_data(image)
            print("[GIF] Saved to {}".format(gif_name))

        # Final result
        if not self.env._complete():
            print("Agents did not reach their goals in max steps.")
        else:
            print("✅ Goodbye world — we did it!")

        return self.env._complete(), step, no_of_collisons, total_rewards,total_on_goal, total_blocking,total_robot_collisions, total_obstacle_collisions


def run_simulations(primal,num_agents,world,goal_world,gif_name):
    
    gym=mapf_gym.MAPFEnv(num_agents=num_agents, world0=world,goals0=goal_world)
    primal.set_env(gym)
    start_time=time.time()
    results=dict()
    start_time=time.time()
    
    try:
        #print('Starting test ({},{},{},{})'.format(n,s,d,id))
        finished, steps_took, collisions, total_rewards, on_goal_count, blocking_count, robot_col_total, obstacle_col_total =primal.find_path(256, gif_name=gif_name)
        success = int(finished)
        results['time']=time.time()-start_time
        
        
    except (OutOfTimeError, NoSolutionError):
        steps_took = 0
        collisions = 0
        total_rewards = 0
        on_goal_count = 0
        blocking_count = 0
        finished = False
        robot_col_total = 0
        obstacle_col_total = 0
        
    time_taken = time.time() - start_time
    collisions=collisions/num_agents
    total_rewards=total_rewards/num_agents
    on_goal_count=on_goal_count/num_agents
    blocking_count=blocking_count/num_agents
    robot_col_total=robot_col_total/num_agents
    obstacle_col_total=obstacle_col_total/num_agents
    results.update({
        'finished': finished,
        'steps': steps_took,
        'collisions': collisions,
        'reward': total_rewards,
        'on_goal': on_goal_count,
        'blocking': blocking_count,
        'robot_collisions': robot_col_total,
        'obstacle_collisions': obstacle_col_total,
        'time': time_taken
    })
    logging.info(
    "[EPISODE {}] Success: {} | Steps: {} || Collisions: {} | Robot: {}| Obstacle: {} ||Reward: {} | On Goal: {} | Blocking: {} | Time: {:.2f}s".format(
        Total_Experiment,
        results['finished'],
        results['steps'],
        results['collisions'],
        results['robot_collisions'],
        results['obstacle_collisions'],
        results['reward'],
        results['on_goal'],
        results['blocking'],
        results['time']
            )
        )
    return finished, success, steps_took, collisions, total_rewards, results
    





if __name__ == "__main__":

    Total_Experiment = 1
    Successful_experiment = 0
    RNN_SIZE = 512
    a_size = 5

    Total_steps = 0
    Total_collision = 0
    Total_Reward = 0
    Total_OnGoal = 0
    Total_Blocking = 0
    Total_Robot_Collisions = 0
    Total_Obstacle_Collisions = 0

    
    primal=PRIMAL('model_primal',10)
   
    env_dir = 'scalability_test_size70x70_agents_50_obs_30_envs'
    pkl_files = sorted([f for f in os.listdir(env_dir) if f.endswith('.pkl')])

    for fname in pkl_files:
        full_path = os.path.join(env_dir, fname)

        # Extract metadata from filename using regex
        match = re.match(r'env_agents_(\d+)_size_(\d+)_prob_([\d.]+)\.pkl', fname)
        if not match:
            print("❌ Skipping unrecognized filename format:", fname)
            continue

        num_agents = int(match.group(1))
        size = int(match.group(2))
        prob = float(match.group(3))

        # print("🧠 Running:", fname)
        # print("🔢 Agents: {}, Size: {}, Prob: {}".format(num_agents, size, prob))
        print("Started experiment run {} with world size {}".format(Total_Experiment, size), flush=True)
        print("Number of agents set to: {} and obstacle prob {}".format(num_agents, prob), flush=True)
        logging.info("Started Experiment {}: Number of agents: {}, World size: {}, Obstacle probability: {:.2f}".format(
            Total_Experiment, num_agents, size, prob))
        # Load environment
        world, goals = load_env_from_pytorch_pickle(full_path)

        # Run simulation
        gif_name = "gifs/{}_rollout.gif".format(os.path.splitext(fname)[0])
        finished, success, steps_count, collisions_count, rewards_count, metrics=run_simulations(primal, num_agents, world, goals, gif_name)
        if finished:
            Successful_experiment += 1

        Total_steps += steps_count
        Total_collision += collisions_count
        Total_Reward += rewards_count 
        Total_OnGoal += metrics['on_goal']
        Total_Blocking += metrics['blocking']
        Total_Robot_Collisions += metrics['robot_collisions']
        Total_Obstacle_Collisions += metrics['obstacle_collisions']
        
        print("✅ Finished:", gif_name)
        Total_Experiment += 1
   

   # Summary logging
    success_rate = (Successful_experiment / (Total_Experiment - 1)) * 100 if Successful_experiment > 0 else 0
    collision_rate = (Total_collision / Total_steps) * 100 if Total_steps > 0 else 0
    avg_steps = Total_steps / (Total_Experiment - 1)
    avg_reward = Total_Reward / Total_steps if Total_steps > 0 else 0
    avg_on_goal = Total_OnGoal / Total_steps if Total_steps > 0 else 0
    avg_blocking = Total_Blocking / Total_steps if Total_steps > 0 else 0
    avg_robot_col = Total_Robot_Collisions / Total_steps if Total_steps > 0 else 0
    avg_obst_col = Total_Obstacle_Collisions / Total_steps if Total_steps > 0 else 0
    logging.info("=" * 80)
    logging.info("🎓 Summary of {} Experiments Completed".format(Total_Experiment - 1))
    logging.info("-" * 80)
    logging.info("✔️ Success Rate       : {:.2f}%".format(success_rate))
    logging.info("💥 Collision Rate     : {:.2f}%".format(collision_rate))
    logging.info("🚶 Avg Steps/Episode   : {:.2f}".format(avg_steps))
    logging.info("🏆 Avg Reward/Step     : {:.4f}".format(avg_reward))
    logging.info("🎯 Avg On-Goal/Step    : {:.4f}".format(avg_on_goal))
    logging.info("🚧 Avg Blocking/Step   : {:.4f}".format(avg_blocking))
    logging.info("🤖 Avg Robot Collisions    : {:.4f}".format(avg_robot_col))
    logging.info("🧱 Avg Obstacle Collisions : {:.4f}".format(avg_obst_col))
    logging.info("=" * 80)

    
print("finished all tests!")

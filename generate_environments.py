import numpy as np
import os
from evaluation_environment import MAPFEnv

def generate_environments(
    save_dir='scalability_test_size70x70_agents_1000_obs_20_envs',
    agent_values=[250,500,750,1000],#range(250, 251, 5),  # 5, 10, ..., 50
    size_list=[70],
    prob_values=[0.0,0.1,0.2]#np.linspace(0.0, 0.1, 7)  # 0.0, 0.05, ..., 0.3
):
    os.makedirs(save_dir, exist_ok=True)

    for num_agents in agent_values:
        for size in size_list:
            for prob in prob_values:
                try:
                    env = MAPFEnv(num_agents=num_agents, SIZE=size, PROB=prob, observation_size=10)
                    env._reset_world()
                    filename = f'env_agents_{num_agents}_size_{size}_prob_{prob:.2f}.pkl'
                    path = os.path.join(save_dir, filename)
                    env.export_env(path)
                    print(f"✅ Saved: {filename}")
                except Exception as e:
                    print(f"❌ Failed for agents={num_agents}, size={size}, prob={prob:.2f}: {e}")

if __name__ == '__main__':
    generate_environments()

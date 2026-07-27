import gc
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
import torch
from io import BytesIO
import os

def visualize_gif(num_agents,grid_size,obs_density,gridworld, paths, start_positions, goal_positions, experiment_no, swap=False):
    print("Visualization of Path Started...")

    # Convert to numpy
    gridworld = gridworld.cpu().numpy().astype(np.uint8)
    num_agents = len(start_positions)
    cmap = plt.get_cmap('tab10')
    agent_colors = [cmap(i % 10) for i in range(num_agents)]
    path_histories = [[] for _ in range(num_agents)]

    gif_filename = (
                f"exp{experiment_no}_agents_{num_agents}_size_{grid_size}_prob_{obs_density:.2f}.gif"
                if not swap else
                f"exp{experiment_no}_agents_{num_agents}_size_{grid_size}_prob_{obs_density:.2f}_swapped.gif"
                    )


    os.makedirs("test_gifs", exist_ok=True)
    gif_filename = os.path.join("test_gifs", gif_filename)

    with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
        for step, positions in enumerate(paths[:256]):
            fig, ax = plt.subplots(figsize=(6, 6))
            height, width = gridworld.shape

            # Set grid settings
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks(np.arange(0, width + 1, 1))
            ax.set_yticks(np.arange(0, height + 1, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color='gray', linewidth=0.5)
            ax.tick_params(left=False, bottom=False)

            # Draw background grid and obstacles
            for x in range(width):
                for y in range(height):
                    color = 'black' if gridworld[y, x] == 1 else 'white'
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black', lw=0.5))

            # Draw start positions (below everything)
            for i, start in enumerate(start_positions):
                sx, sy = start[1], start[0]
                ax.add_patch(Rectangle(
                    (sx, sy), 1, 1,
                    facecolor=agent_colors[i],
                    edgecolor='black',
                    lw=1.5,
                    alpha=0.2
                ))
                ax.plot(sx + 0.5, sy + 0.5, marker='o', color=agent_colors[i], markersize=6, markeredgecolor='black')

            # Draw goal positions
            for i, goal in enumerate(goal_positions):
                gx, gy = goal[1], goal[0]
                ax.add_patch(Rectangle(
                    (gx, gy), 1, 1,
                    facecolor=agent_colors[i],
                    edgecolor='black',
                    lw=1.5,
                    alpha=0.3
                ))
                ax.plot(gx + 0.5, gy + 0.5, marker='*', color=agent_colors[i], markersize=8, markeredgecolor='black')

            # Draw path histories and current agent positions
            for i, pos in enumerate(positions):
                path_histories[i].append(pos)

                # Path lines
                if len(path_histories[i]) > 1:
                    x_coords = [p[1] + 0.5 for p in path_histories[i]]
                    y_coords = [p[0] + 0.5 for p in path_histories[i]]
                    ax.plot(x_coords, y_coords, color=agent_colors[i], lw=1.2, alpha=0.5)

                # Current agent square
                ax.add_patch(Rectangle((pos[1], pos[0]), 1, 1, facecolor=agent_colors[i], edgecolor='black', lw=1))

            ax.set_title(f"Time Step: {step}", fontsize=10)

            # Save to GIF frame
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            writer.append_data(imageio.imread(buf))
            buf.close()
            plt.close(fig)
            gc.collect()

    print(f"✅ GIF saved to {gif_filename}")

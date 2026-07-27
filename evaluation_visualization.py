import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
import os, gc

def visualize_gif(
    gridworld, paths, start_positions, goal_positions,
    experiment_no, grid_size, obs_density,agents_completed=None, total_agents=None, swap=False, fps=10, max_steps=None
):
    print("Visualization of Path Started...")

    gridworld = gridworld.cpu().numpy().astype(np.uint8)
    num_agents = len(start_positions)
    height, width = gridworld.shape

    # Colors
    if num_agents <= 20:
        cmap = plt.get_cmap('tab20')
        agent_colors = [cmap(i) for i in range(num_agents)]
    else:
        agent_colors = [plt.cm.hsv(i / num_agents) for i in range(num_agents)]

    path_histories = [[] for _ in range(num_agents)]

    os.makedirs("test_videos", exist_ok=True)
    mp4_filename = (
        f"exp{experiment_no}_agents_{num_agents}_size_{grid_size}_prob_{obs_density:.2f}.mp4"
        if not swap else
        f"exp{experiment_no}_agents_{num_agents}_size_{grid_size}_prob_{obs_density:.2f}_swapped.mp4"
    )
    mp4_filename = os.path.join("test_videos", mp4_filename)

    writer = imageio.get_writer(
        mp4_filename, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )

    total_steps = len(paths)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    # Fade duration (first 60% of total steps)
    fade_steps = int(total_steps * 0.6)

    try:
        for step in range(total_steps):
            positions = paths[step]

            # Compute fade alpha for start positions
            fade = max(0.0, 1.0 - step / fade_steps) if step < fade_steps else 0.0

            fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks(np.arange(0, width + 1, 1))
            ax.set_yticks(np.arange(0, height + 1, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color='gray', linewidth=0.5, alpha=0.4)
            ax.tick_params(left=False, bottom=False)

            # Background & obstacles
            for x in range(width):
                for y in range(height):
                    color = 'black' if gridworld[y, x] == 1 else 'white'
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black', lw=0.5))

            # --- Start markers (fade out gradually) ---
            for i, (r, c) in enumerate(start_positions):
                ax.add_patch(Rectangle((c, r), 1, 1,
                                       facecolor=agent_colors[i],
                                       edgecolor='black', lw=1.0,
                                       alpha=0.3 * fade))
                ax.plot(c + 0.5, r + 0.5, marker='o',
                        color=agent_colors[i], markersize=6,
                        markeredgecolor='black', alpha=0.8 * fade)

            # Goal markers
            for i, (r, c) in enumerate(goal_positions):
                ax.add_patch(Rectangle((c, r), 1, 1,
                                       facecolor=agent_colors[i],
                                       edgecolor='black', lw=1.5, alpha=0.3))
                ax.plot(c + 0.5, r + 0.5, marker='*',
                        color=agent_colors[i], markersize=8,
                        markeredgecolor='black')

            # Paths + current positions
            for i, pos in enumerate(positions):
                path_histories[i].append(pos)
                ax.add_patch(Rectangle((pos[1], pos[0]), 1, 1,
                                       facecolor=agent_colors[i],
                                       edgecolor='black', lw=1))

            # ax.set_title(f"Time Step: {step}", fontsize=10)
            title = f"Step: {step}"

            # Only show completion info on final frame
            if (
                agents_completed is not None
                and total_agents is not None
                and step == total_steps - 1
            ):
                title += f" | Agents Completed: {agents_completed}/{total_agents}"

            ax.set_title(title, fontsize=10)


            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            frame = buf.reshape((h, w, 3))
            writer.append_data(frame)

            plt.close(fig)
            gc.collect()
    finally:
        writer.close()

    print(f"✅ Video saved to {mp4_filename}")

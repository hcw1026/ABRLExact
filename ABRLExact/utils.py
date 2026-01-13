import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
import numpy as np
from collections import defaultdict

from matplotlib.animation import FuncAnimation

import torch

from IPython.display import HTML  # Import this for notebook display


def get_default_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu")


def sample_path_with_prob(all_paths, path_probs):#, deterministic_path=True):
    path_number = np.random.choice(len(all_paths), size=1, p=path_probs)[0]
    policy = {k: v for (k,v) in all_paths[path_number]}
    return policy, path_probs


def compute_q_optimal_probs(samples, all_paths, env):
    samples_optim_prob = np.zeros(len(all_paths))
    for p, path in enumerate(all_paths):
        samples_mask = np.zeros(len(samples), dtype=np.bool)
        for num, sample in enumerate(samples):
            mask = True
            for state, action in path:
                mask = mask & np.all(sample[env.nu(state, action)-1] > sample[env.nu(state, 1-action)-1])
            samples_mask[num] = mask
        samples_optim_prob[p] = np.mean(samples_mask)
    return samples_optim_prob


def add_obs(obs, unique_stat, state0, action, state1, reward, done):
    tr_id = (*state0, action)
    added_flag = False
    if tr_id in unique_stat:
        return obs, unique_stat, added_flag
    else:
        unique_stat.add(tr_id)
        obs["state0"].append(state0)
        obs["action"].append(action)
        obs["state1"].append(state1)
        obs["rewards"].append(reward)
        obs["done"].append(done)
        added_flag = True
        return obs, unique_stat, added_flag


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def draw_entropy_node(r, c, ax, radius, cmap, probs, pyramid=False):
    if not pyramid:
        centre = (c, r)
    else:
        centre = (c - 0.5*r, r)
    
    if probs is not None:
        idx = get_flattened_index(r, c)
        if idx < len(probs):
            p = probs[idx]
            # entropy = binary_entropy(p)
            entropy = (np.maximum(p, 1-p) - 0.5) * 2
            fill_color = cmap(entropy)
        else:
            fill_color="#808080"
    else:
        fill_color = "#FFFFFF"


    
    circle = plt.Circle(centre, radius, facecolor=fill_color, edgecolor='black', linewidth=1, zorder=20)
    ax.add_patch(circle)

def draw_pie_node(r, c, ax, radius, probs, action_swap=False, pyramid=False):
    if not pyramid:
        centre = (c, r)
    else:
        centre = (c - 0.5*r, r)
    
    color_left = '#EF9A9A'
    color_right = '#90CAF9'

    if probs is not None:
        idx = get_flattened_index(r, c)
        if idx < len(probs):
            p = probs[idx]
            p = min(1., p)
            p = max(0., p)
            p = p if action_swap is False else 1-p
    
            start_angle = 90.0 # clockwise due to inverted y-axis
    
            theta1_p = start_angle
            theta2_p = start_angle + (360 * (1-p))
            
            wedge_p = patches.Wedge(centre, radius, theta1_p, theta2_p,
                                    facecolor=color_left, edgecolor='black', linewidth=0.5, zorder=20)
            ax.add_patch(wedge_p)
            
            wedge_rest = patches.Wedge(centre, radius, theta2_p, theta1_p + 360,
                                       facecolor=color_right, edgecolor='black', linewidth=0.5, zorder=20)
            ax.add_patch(wedge_rest)
        else:
            blank_color = "#808080"
            circle = circle = plt.Circle(centre, radius, facecolor=blank_color, edgecolor='black', linewidth=1, zorder=20)
            ax.add_patch(circle)
    else:
        blank_color = "#FFFFFF"
        circle = circle = plt.Circle(centre, radius, facecolor=blank_color, edgecolor='black', linewidth=1, zorder=20)
        ax.add_patch(circle)

def get_arrow_coords(start_node, end_node, radius, pyramid=False):
    r1, c1 = start_node
    r2, c2 = end_node

    if pyramid:
        c1 -= 0.5 * r1
        c2 -= 0.5 * r2

    diag_offset = radius * (2**0.5) / 2 
    
    if c2 > c1:
        return (c1 + diag_offset, r1 + diag_offset), (c2 - diag_offset, r2 - diag_offset)
    elif c2 == c1:
        return (c1, r1 + radius), (c2, r2 - radius)
    else:
        return (c1 - diag_offset, r1 + diag_offset), (c2 + diag_offset, r2 - diag_offset)

def draw_arrow(start_node, end_node, ax, radius, color, style='-', alpha=1.0, width=1.0, zorder=1, pyramid=False):
    s_pos, e_pos = get_arrow_coords(start_node, end_node, radius=radius, pyramid=pyramid)
    
    lw = 1.0 if style != '-' else 0
        
    arrow = patches.FancyArrowPatch(
        s_pos, e_pos,
        arrowstyle=f"Simple, tail_width={width}, head_width={width*3}, head_length={width*3}",
        alpha=alpha, color=color, linestyle=style, linewidth=lw, zorder=zorder,
        shrinkA=0, shrinkB=0
    )
    ax.add_patch(arrow)

def get_flattened_index(r, c):
    return int((r * (r + 1)) / 2) + c

def plot_deepsea(trajectories, depth, probs=None, action_map=None, draw_background=False, draw_curr_traj=True, sim_traj_dict=None, pyramid=False):
    fig, ax = plt.subplots(figsize=(int(depth*1.7), int(depth*1.7)))

    # colorbar
    norm = colors.Normalize(vmin=0, vmax=1)
    cmap_entropy = plt.get_cmap('Spectral_r')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_entropy)
    sm.set_array([])

    ax = render_deepsea(ax=ax, cmap=cmap_entropy, trajectories=trajectories, frame_index=None, depth=depth, 
                        probs=probs, action_map=action_map, sim_traj_dict=sim_traj_dict, draw_curr_traj=draw_curr_traj, draw_background=draw_background, pyramid=pyramid)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.05)
    cbar.set_label('Optimal Action Entropy', rotation=270, labelpad=15, fontsize=12)

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()


def render_deepsea(ax, trajectories, frame_index, depth, probs=None, cmap=None, action_map=None, mode="entropy", sim_traj_dict=None, draw_curr_traj=True, draw_background=False, pyramid=False):
    
    radius = 0.25

    if pyramid is False:
        ax.set_xlim(-0.5, depth - 0.5)
    else:
        ax.set_xlim(- depth / 2 - 0.5, depth / 2 + 0.5)
    ax.set_ylim(depth - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    if frame_index is not None:
        ax.set_title(f"Frame: {frame_index}", fontsize=14, y=0.95)

    # draw nodes
    for r in range(depth):
        for c in range(r + 1):
            if mode == "entropy":
                draw_entropy_node(r, c, ax=ax, radius=radius, probs=probs, cmap=cmap, pyramid=pyramid)
            elif mode == "pie":
                action_swap = True if (action_map is not None) and (r < depth-1) and (action_map[r,c] == 0) else False
                draw_pie_node(r, c, ax=ax, radius=radius, probs=probs, action_swap=action_swap, pyramid=pyramid)
            else:
                raise ValueError("undefined mode")

    # draw past trajectories
    if sim_traj_dict is not None and sim_traj_dict["num_sims"] > 0:
        plot_trajectories = []
        all_paths = sim_traj_dict["all_paths"]
        path_probs = sim_traj_dict["path_probs"][frame_index]
        env = sim_traj_dict["env"]
        num_sims = sim_traj_dict["num_sims"]
    
        for i in range(num_sims):
            policy, path_probs = sample_path_with_prob(all_paths, path_probs)#, deterministic_path=env.deterministic_transition)
            curr_state = (0,0)
            history = [curr_state]
            done = False
            while done is False:
                action = policy[curr_state] #policy[d] if env.deterministic_transition else policy[curr_state]
                next_state, _, done = env.step(action=action, is_simulation=True, simulation_state=curr_state)
                curr_state = next_state
                if env.deterministic_transition and next_state in history:
                    break
                history.append(next_state)
            plot_trajectories.append(history)

        edge_counts = defaultdict(int)
        
        for traj in plot_trajectories:
            for t in range(len(traj) - 1):
                edge = (traj[t], traj[t+1])
                edge_counts[edge] += 1
        
        for edge, count in edge_counts.items():
            alpha = min(1.0, count / num_sims)
            draw_arrow(edge[0], edge[1], ax=ax, color='red', radius=radius, width=2.0, alpha=alpha, zorder=5, pyramid=pyramid)
            
    else:    
        edge_counts = defaultdict(int)
        
        for i in range(frame_index):
            traj = trajectories[i]
            for t in range(len(traj) - 1):
                edge = (traj[t], traj[t+1])
                edge_counts[edge] += 1

            max_visit = 5.0
            
        for edge, count in edge_counts.items():
            alpha = min(1.0, count / max_visit)
            draw_arrow(edge[0], edge[1], ax=ax, color='black', radius=radius, width=1.2, alpha=alpha, zorder=5, pyramid=pyramid) 
    

    # draw background
    if draw_background:
        bg_color = "#BDBDBD"
        for r in range(depth - 1):
            for c in range(r + 1):
                curr = (r, c)
                if not pyramid:
                    if edge_counts[(curr, (r+1, c+1))] == 0:
                        draw_arrow(curr, (r+1, c+1), ax=ax, color=bg_color, radius=radius, style=(0, (1, 4)), width=0.3, pyramid=pyramid)
        
                    target_c = 0 if c == 0 else c - 1
                    if edge_counts[(curr, (r+1, target_c))] == 0:
                        draw_arrow(curr, (r+1, target_c), ax=ax, color=bg_color, radius=radius, style=(0, (1, 4)), width=0.3, pyramid=pyramid)
                else:
    
                    if edge_counts[(curr, (r+1, c))] == 0:
                        draw_arrow(curr, (r+1, c), ax=ax, color=bg_color, radius=radius, style=(0, (1, 4)), width=0.3, pyramid=pyramid)
                    if edge_counts[(curr, (r+1, c+1))] == 0:
                        draw_arrow(curr, (r+1, c+1), ax=ax, color=bg_color, radius=radius, style=(0, (1, 4)), width=0.3, pyramid=pyramid)

    # draw current trajectory
    if draw_curr_traj and frame_index < len(trajectories):
        current_traj = trajectories[frame_index]
        for t in range(len(current_traj) - 1):
            draw_arrow(current_traj[t], current_traj[t+1], ax=ax, color='green', radius=radius, width=2.0, zorder=10, pyramid=pyramid)

    return ax

def animate_deepsea_entropy_trajectories(trajectories, probs=None, action_map=None, filename="deepsea_clean.gif", save=False, interval=200, 
                                         sim_traj_dict=None, draw_curr_traj=True, draw_background=False, pyramid=False, depth=None):
    """
    Creates an animation by calling render_deepsea_frame repeatedly.
    """
    if depth is None:
        assert len(set([len(t) for t in trajectories[1:]])) == 1
        depth = len(trajectories[-1])
    else:
        depth += 1
    
    fig, ax = plt.subplots(figsize=(8, 7))

    colors_list = [
        (0.00, '#0000FF'),
        (0.05, '#00FFFF'),
        (0.15, '#CCFF00'),
        (0.50, '#FFFF00'),
        (0.85, '#FF8000'),
        (0.95, '#FF4000'),
        (1.00, '#FF0000')]
    cmap_entropy = colors.LinearSegmentedColormap.from_list('EntropyFocus', colors_list, N=256)

    # colorbar
    norm = colors.Normalize(vmin=0, vmax=1)
    #cmap_entropy = plt.get_cmap('Spectral_r')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_entropy)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.05)
    cbar.set_label('Optimal Action Entropy', rotation=270, labelpad=15, fontsize=12)

    mode = "entropy"

    def update(frame):
        ax.clear()
        render_deepsea(ax=ax, cmap=cmap_entropy, trajectories=trajectories, mode=mode, frame_index=frame, probs=probs[frame], 
                       action_map=action_map, depth=depth, sim_traj_dict=sim_traj_dict, draw_curr_traj=draw_curr_traj, draw_background=draw_background, pyramid=pyramid)

    # 3. Animate
    ani = FuncAnimation(fig, update, frames=len(trajectories), blit=False, repeat=False, interval=interval)

    if save:
        ani.save(filename, writer='pillow', fps=5)
        print(f"Animation saved to {filename}")

    plt.close()
    return ani


def animate_deepsea_pie_trajectories(trajectories, probs=None, action_map=None, filename="deepsea_clean.gif", save=False, interval=200, 
                                     sim_traj_dict=None, draw_curr_traj=True, draw_background=False, pyramid=False, depth=None):
    """
    Creates an animation by calling render_deepsea_frame repeatedly.
    """
    if depth is None:
        assert len(set([len(t) for t in trajectories[1:]])) == 1
        depth = len(trajectories[-1])
    else:
        depth += 1
    
    fig, ax = plt.subplots(figsize=(8, 7))

    def update(frame):
        ax.clear()
        render_deepsea(ax=ax, trajectories=trajectories, frame_index=frame, probs=probs[frame], action_map=action_map, depth=depth, mode="pie", 
                       sim_traj_dict=sim_traj_dict, draw_curr_traj=draw_curr_traj, draw_background=draw_background, pyramid=pyramid)

    # 3. Animate
    ani = FuncAnimation(fig, update, frames=len(trajectories), blit=False, repeat=False, interval=interval)

    if save:
        ani.save(filename, writer='pillow', fps=5)
        print(f"Animation saved to {filename}")

    plt.close()
    return ani


def animate_trajectories(trajectories, filename="trajectory_animation.gif", save=False, interval=200):
    """
    Animates trajectories, swaps x and y, top-left origin, square grid, frame number, and y labels at the top.
    """

    n = len(trajectories[0])

    fig, ax = plt.subplots()
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(0, n + 1, 1))
    ax.set_yticks(np.arange(0, n + 1, 1))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    ax.tick_params(axis='y', top=True, labeltop=True, bottom=False, labelbottom=False)

    lines = []
    for _ in trajectories:
        line, = ax.plot([], [], 'r-')
        lines.append(line)

    frame_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top')

    def init():
        for line in lines:
            line.set_data([], [])
        frame_text.set_text('')
        return tuple(lines + [frame_text]) # return tuple

    def update(frame):
        if frame < len(trajectories):
            trajectory = trajectories[frame]
            y_coords = [coord[0] + 0.5 for coord in trajectory]
            x_coords = [coord[1] + 0.5 for coord in trajectory]
            lines[frame].set_data(x_coords, y_coords)

            for i in range(frame):
                lines[i].set_color('gray')
                lines[i].set_alpha(0.5)

            frame_text.set_text(f"Frame: {frame}")
        return tuple(lines + [frame_text]) # return tuple

    ani = FuncAnimation(fig, update, frames=len(trajectories) + 1, init_func=init, blit=True, repeat=False, interval=interval)

    if save:
        ani.save(filename, writer='pillow', fps=2)
        
    plt.close() 
    
    return ani
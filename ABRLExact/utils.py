from collections import defaultdict
import itertools
import math
import os

from matplotlib import ticker
from matplotlib.animation import FuncAnimation
import matplotlib.cm as mcm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch

from omegaconf import OmegaConf


from ABRLExact.environment import DeepSea, DeepSeaPyramid, DeepSeaSwirl

cm = 1/2.54


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
    
    
def get_all_deterministic_paths(env):
    num_actions = len(env.get_all_states()) - len(env.get_all_terminal_states())
    all_action_combinations = list(itertools.product([0, 1], repeat=num_actions))
    state_id = {state: num for num, state in enumerate(env.get_all_states())}
    path_set = set()
    for comb in all_action_combinations:
        path = []
        curr_state = (0, 0)
        done = False
        history = [curr_state]
        while done is False:
            action = comb[state_id[curr_state]]
            next_state, _, done = env.step(action, is_simulation=True, simulation_state=curr_state)
            path.append((curr_state, action))
            if next_state in history:
                break
            history.append(next_state)
            curr_state = next_state
        path_set.add((tuple(path), tuple(history)))
    all_paths_histories = list(path_set)
    all_paths = [list(ph[0]) for ph in all_paths_histories]
    all_histories = [list(ph[1]) for ph in all_paths_histories]
    return all_paths, all_histories


def get_all_stochastic_paths(env):
    all_states = []
    for state in env.get_all_states():
        if state not in env.get_all_terminal_states():
            all_states.append(state)

    all_paths_choices = [[(s, a) for a in env.get_possible_actions(state=s, gym_space=False)] for s in all_states]
    all_paths = list(itertools.product(*all_paths_choices))
    return all_paths


def find_existing_run(base_dir, config): # pylint: disable=redefined-outer-name
    if not os.path.exists(base_dir):
        return None
    
    target_config = config.copy()
    target_config.pop("num_experiments", None)
    is_cdf_bs1 = target_config.get("method", "").lower() == "cdf_bs1"
    if is_cdf_bs1:
        target_config.pop("num_bs_samples", None)

    for entry in os.scandir(base_dir):
        if entry.is_dir():
            config_path = os.path.join(entry.path, "config.yaml")
            if os.path.exists(config_path):
                existing_config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
                existing_config.pop("num_experiments", None)
                if is_cdf_bs1:
                    existing_config.pop("num_bs_samples", None)
                if existing_config == target_config:
                    return entry.path
    return None


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
            p = np.round(probs[idx], 2)
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
            circle = circle = plt.Circle(centre, radius, facecolor=blank_color, edgecolor='black', linewidth=0.5, zorder=20)
            ax.add_patch(circle)
    else:
        blank_color = "#FFFFFF"
        circle = circle = plt.Circle(centre, radius, facecolor=blank_color, edgecolor='black', linewidth=0.5, zorder=20)
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


def draw_arrow(start_node, end_node, ax, radius, color, style='-', alpha=1.0, width=1.0, zorder=1, arrow_style=None, linewidth=None, fill=True, pyramid=False):
    s_pos, e_pos = get_arrow_coords(start_node, end_node, radius=radius, pyramid=pyramid)

    if arrow_style is None:
        final_arrow_style = f"Simple, tail_width={width}, head_width={width*3}, head_length={width*3}"
        
        if linewidth is not None:
            lw = linewidth
        elif not fill:
            lw = 1.5
        else:
            lw = 1.0 if style != '-' else 0
    else:
        final_arrow_style = arrow_style
        lw = width if linewidth is None else linewidth

        
    arrow = patches.FancyArrowPatch(
        s_pos, e_pos,
        arrowstyle=final_arrow_style,
        alpha=alpha, color=color, linestyle=style, linewidth=lw, zorder=zorder,
        shrinkA=0, shrinkB=0, fill=fill
    )
    ax.add_patch(arrow)


def get_flattened_index(r, c):
    return int((r * (r + 1)) / 2) + c


def render_deepsea(ax, trajectories, frame_index, depth, probs=None, cmap=None, action_map=None, mode="entropy", sim_traj_dict=None, 
                   draw_curr_traj=True, draw_unexplored=False, draw_background=False, pyramid=False, env=None, obs_history=None):
    
    radius = 0.25

    if pyramid is False:
        ax.set_xlim(-0.5, depth - 0.5)
    else:
        ax.set_xlim(- depth / 2 - 0.5, depth / 2 + 0.5)
    ax.set_ylim(depth - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    if frame_index is not None:
        ax.set_title(f"Episode: {frame_index}", y=0.95)

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
            draw_arrow(edge[0], edge[1], ax=ax, color='#cc5500', radius=radius, width=2.0, alpha=alpha, zorder=5, pyramid=pyramid)
            
    else:    
        edge_counts = defaultdict(int)
        
        for i in range(frame_index+1):
            traj = trajectories[i]
            for t in range(len(traj) - 1):
                edge = (traj[t], traj[t+1])
                edge_counts[edge] += 1

            max_visit = 5.0
            
        for edge, count in edge_counts.items():
            alpha = min(1.0, count / max_visit)
            draw_arrow(edge[0], edge[1], ax=ax, color='black', radius=radius, width=1.2, alpha=alpha, zorder=5, pyramid=pyramid) 

    # draw unexplored
    if draw_unexplored:
        edge_explored = defaultdict(int)
        if env.deterministic_transition is True:
            for i in range(frame_index+1):
                traj = trajectories[i]
                for t in range(len(traj) - 1):
                    edge = (traj[t], traj[t+1])
                    edge_explored[edge] = 1
        else:
            for i in range(frame_index+1):
                obs = obs_history[i]
                for t in range(len(obs["state0"])):
                    edge = (tuple(obs["state0"][t]), tuple(env.trans_info[obs["state0"][t] + (obs["action"][t], )]))
                    edge_explored[edge] = 1

        bg_color = "#BDBDBD"
        for r in range(depth - 1):
            for c in range(r + 1):
                curr = (r, c)
                next_state = tuple(env.trans_info[curr + (0, )])
                if edge_explored[(curr, next_state)] == 0:
                    draw_arrow(curr, next_state, ax=ax, color=bg_color, radius=radius, style=":", arrow_style="-", linewidth=1, pyramid=pyramid)
    
                next_state = tuple(env.trans_info[curr + (1, )])
                if edge_explored[(curr, next_state)] == 0:
                    draw_arrow(curr, next_state, ax=ax, color=bg_color, radius=radius, style=":", arrow_style="-", linewidth=1, pyramid=pyramid)
        

    # draw background
    if draw_background:
        bg_color = "#BDBDBD"
        for r in range(depth - 1):
            for c in range(r + 1):
                curr = (r, c)
                next_state = tuple(env.trans_info[curr + (0, )])
                if edge_counts[(curr, next_state)] == 0:
                    draw_arrow(curr, next_state, ax=ax, color=bg_color, radius=radius, width=2.0, pyramid=pyramid)
    
                next_state = tuple(env.trans_info[curr + (1, )])
                if edge_counts[(curr, next_state)] == 0:
                    draw_arrow(curr, next_state, ax=ax, color=bg_color, radius=radius, width=2.0, pyramid=pyramid)


    # draw current trajectory
    # if draw_curr_traj and frame_index + 1 < len(trajectories):
    #     current_traj = trajectories[frame_index+1]
    #     for t in range(len(current_traj) - 1):
    #         draw_arrow(current_traj[t], current_traj[t+1], ax=ax, color='green', radius=radius, width=1.2, zorder=20, pyramid=pyramid)
    if draw_curr_traj and frame_index + 1 < len(trajectories):
        current_traj = trajectories[frame_index+1]
        
        is_sim_mode = (sim_traj_dict is not None and sim_traj_dict["num_sims"] > 0)
        
        for t in range(len(current_traj) - 1):
            if is_sim_mode:
                draw_arrow(current_traj[t], current_traj[t+1], ax=ax, color='black', 
                           radius=radius, width=2, linewidth=0.5, zorder=20, pyramid=pyramid, fill=False)
            else:
                draw_arrow(current_traj[t], current_traj[t+1], ax=ax, color='green', 
                           radius=radius, width=1, zorder=20, pyramid=pyramid)

    return ax


def animate_deepsea_pie_trajectories(trajectories, probs=None, action_map=None, save_path=None, interval=200, 
                                     sim_traj_dict=None, draw_curr_traj=True, draw_background=False, draw_unexplored=False, 
                                     pyramid=False, depth=None, env=None, obs_history=None):
    """Creates an animation by calling render_deepsea_frame repeatedly."""
    if depth is None:
        assert len(set([len(t) for t in trajectories[1:]])) == 1
        depth = len(trajectories[-1])
    else:
        depth += 1
    
    fig, ax = plt.subplots(figsize=(8, 7))

    def update(frame):
        ax.clear()
        render_deepsea(ax=ax, trajectories=trajectories, frame_index=frame, probs=probs[frame], action_map=action_map, depth=depth, mode="pie", 
                       sim_traj_dict=sim_traj_dict, draw_curr_traj=draw_curr_traj, draw_unexplored=draw_unexplored, 
                       draw_background=draw_background, pyramid=pyramid, env=env, obs_history=obs_history)

    # 3. Animate
    ani = FuncAnimation(fig, update, frames=len(trajectories), blit=False, repeat=False, interval=interval)

    if save_path:
        ani.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")

    plt.close()
    return ani


def plot_cumulative_regret(regret_plots, method_names, ours_labels, epsilon_list, figsize_per_plot=(7.5,6), xtick_freq=None, save_path=None):

    if not isinstance(regret_plots, list):
        regret_plots = [regret_plots]
    
    if isinstance(method_names, str):
        method_names = [method_names]

    num_methods = len(regret_plots)

    if num_methods == 1:
        n_plots = 1
        is_comparison = False
    else:
        n_plots = num_methods - 1
        is_comparison = True

    if n_plots > 4:
        print(f"Warning: Provided {n_plots} plots. Only the first four are plotted")
        n_plots = 4

    fig, axes = plt.subplots(1, n_plots, figsize=(figsize_per_plot[0]*cm*n_plots, figsize_per_plot[1]*cm), sharex=True, sharey=True)

    if n_plots == 1:
        axes = [axes]
    
    colors_list = plt.cm.viridis(np.linspace(0, 0.9, len(epsilon_list)))
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    for idx, eps in enumerate(epsilon_list):
        current_color = colors_list[idx]
        
        base_cum = np.cumsum(regret_plots[0][idx], axis=1)
        base_mean = np.mean(base_cum, axis=0)
        base_std = np.std(base_cum, axis=0)
        episodes = np.arange(len(base_mean))

        for i in range(n_plots):
            ax = axes[i]
            
            ax.plot(episodes, base_mean, color=current_color, linestyle=styles[0], linewidth=1)
            ax.fill_between(episodes, base_mean - base_std, base_mean + base_std, 
                            color=current_color, alpha=0.15, linestyle=styles[0])

            if is_comparison:
                comp_idx = i + 1
                
                comp_cum = np.cumsum(regret_plots[comp_idx][idx], axis=1)
                comp_mean = np.mean(comp_cum, axis=0)
                comp_std = np.std(comp_cum, axis=0)
                    
                curr_style = styles[comp_idx]

                ax.plot(episodes, comp_mean, color=current_color, linestyle=curr_style, linewidth=1)
                ax.fill_between(episodes, comp_mean - comp_std, comp_mean + comp_std, 
                                color=current_color, alpha=0.15, linestyle=curr_style)
                
                ax.set_title(f"{method_names[0]} vs {method_names[comp_idx]}")
            else:
                ax.set_title(f"{method_names[0]}")


    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(0,  max([len(r[0][0]) for r in regret_plots]) + 1)
        if xtick_freq is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_freq))
        ax.tick_params(labelleft=True)

    fig.supylabel("cumulative regret", x=0.05, y=0.6)
    fig.supxlabel("episode", y=0.18)

    sns.despine()

    # legend
    method_handles = []
    ours_label = " (ours)" if isinstance(ours_labels, list) and 0 in ours_labels else ""
    method_handles.append(Line2D([0], [0], color='black', lw=1.5, linestyle=styles[0], label=method_names[0] + ours_label))

    if is_comparison:
        for i in range(n_plots):
            comp_idx = i + 1
            curr_style = styles[comp_idx]
            ours_label = " (ours)" if isinstance(ours_labels, list) and comp_idx in ours_labels else ""
            method_handles.append(Line2D([0], [0], color='black', lw=1.5, linestyle=curr_style, label=method_names[comp_idx] + ours_label))

    plt.subplots_adjust(bottom=0.35, right=0.98, wspace=0.25)

    fig.legend(
        handles=method_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.07),
        ncol=len(method_handles),
        frameon=False,
        handlelength=1.5,
    )

    epsilon_handles = [Line2D([0], [0], color=colors_list[i], lw=2, label=rf'$\epsilon={eps}$') for i, eps in enumerate(epsilon_list)]
    
    fig.legend(
        handles=epsilon_handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(epsilon_list), 
        frameon=False,
        handlelength=1.5
    )


    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_acc_prob(acc_list, epsilon_list, figsize=(10,8), colors_list=None, plot_legend=True, xtick_freq=None, save_path=None):

    _, ax = plt.subplots(figsize=(figsize[0]*cm, figsize[1]*cm))
    
    colors_list = plt.cm.viridis(np.linspace(0, 0.9, len(epsilon_list))) if colors_list is None else colors_list

    for idx, eps in enumerate(epsilon_list):
        current_color = colors_list[idx]

        mean = np.mean(acc_list[idx], axis=0)
        std = np.std(acc_list[idx], axis=0)
        
        episodes = np.arange(len(mean))

        ax.plot(episodes, mean, label=rf"$\epsilon={eps}$", 
                color=current_color, linestyle="-", linewidth=0.5)
        ax.fill_between(episodes, mean - std, mean + std, 
                        color=current_color, alpha=0.15)

    ax.set_xlim(0, max(*[len(l[0]) for l in acc_list]) + 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if xtick_freq is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_freq))
    
    ax.set_xlabel("episode")
    ax.set_ylabel("acceptance prob.")
    ax.grid(True, linestyle=':', alpha=0.6)

    if plot_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title="tolerance", handlelength=1.0)
    sns.despine()

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_optimal_probs_single(all_paths, optimal_paths_rewards, optimal_probs_array, epsilon, figsize=(10,8), plot_error=False, save_path=None):
    fig, ax = plt.subplots(figsize=(figsize[0]*cm, figsize[1]*cm))

    ax.set_xlabel("episode")
    ax.set_ylabel("path opitmal probability")

    sm = plot_optimal_probs_subplot(ax=ax, 
                                    all_paths=all_paths, 
                                    optimal_paths_rewards=optimal_paths_rewards, 
                                    optimal_probs_array=optimal_probs_array, 
                                    title = rf"$\epsilon={epsilon}$",
                                    plot_error=plot_error)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('path total reward', rotation=270, labelpad=15)
    
    sns.despine()

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_optimal_probs_subplot(ax, all_paths, optimal_paths_rewards, optimal_probs_array, title, plot_error=False):

    norm = colors.Normalize(vmin=np.min(optimal_paths_rewards), 
                             vmax=np.max(optimal_paths_rewards))
    cmap = plt.get_cmap('coolwarm')
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)

    for path_idx, _ in enumerate(all_paths):
        reward = optimal_paths_rewards[path_idx]
        current_color = sm.to_rgba(reward)

        mean = np.mean(optimal_probs_array[:,:,path_idx], axis=0)
        std = np.std(optimal_probs_array[:,:,path_idx], axis=0)

        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, color=current_color, linestyle="-", linewidth=1, alpha=0.8)
        if plot_error:
            ax.fill_between(episodes, np.clip(mean - std, 0., 1.), np.clip(mean + std, 0., 1.), color=current_color, edgecolor="none", alpha=0.1)


    ax.set_title(title)
    ax.set_xlim(0, optimal_probs_array.shape[1] + 1)
    ax.set_ylim(-0.02,1.02)
    ax.grid(True, linestyle=':', alpha=0.6)

    return sm


def plot_optimal_probs(all_paths, optimal_paths_rewards, optimal_probs_array_cdf, optimal_probs_array_hmc, epsilon_list, figsize=(15,9), plot_error=False, save_path=None):
    fig, axes = plt.subplots(2, len(epsilon_list), figsize=(figsize[0]*cm, figsize[1]*cm), sharex=True, sharey=True)
    
    sm = None
    for row_idx in range(2):
        for col_idx in range(len(epsilon_list)):
            ax = axes[row_idx][col_idx]
    
            sm = plot_optimal_probs_subplot(ax=ax, 
                                            all_paths=all_paths, 
                                            optimal_paths_rewards=optimal_paths_rewards, 
                                            optimal_probs_array=optimal_probs_array_cdf[col_idx] if row_idx == 0 else optimal_probs_array_hmc[col_idx], 
                                            title = rf"$\epsilon={epsilon_list[col_idx]}$" if row_idx == 0 else None,
                                            plot_error=plot_error)

            if col_idx == 0 and row_idx == 0:
                ax.set_ylabel("policy\nprobability (BR)")
            if col_idx == 0 and row_idx == 1:
                ax.set_ylabel("policy\nprobability (HMC)")
    
            # if row_idx == 1:
            #     ax.set_xlabel("Episode")
            if col_idx > 0:
                ticks = ax.xaxis.get_major_ticks()
                if len(ticks) > 0:
                    ticks[0].label1.set_visible(False)

    sns.despine()

    fig.supxlabel("episode", y=0.02)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('path cumulative reward', rotation=270, labelpad=15)

    plt.subplots_adjust(right=0.9, bottom=0.15, wspace=0.25, hspace=0.15)

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_optimal_probs_hmc_add(all_paths, optimal_paths_rewards, optimal_probs_array_cdf,
                               epsilon_list, figsize=(15,9), plot_error=False, save_path=None):
    fig, axes = plt.subplots(1, len(epsilon_list), figsize=(figsize[0]*cm, figsize[1]*cm), sharex=True, sharey=True)
    
    sm = None
    for idx in range(len(epsilon_list)):
        ax = axes[idx]

        sm = plot_optimal_probs_subplot(ax=ax, 
                                        all_paths=all_paths, 
                                        optimal_paths_rewards=optimal_paths_rewards, 
                                        optimal_probs_array=optimal_probs_array_cdf[idx], 
                                        title = rf"$\epsilon={epsilon_list[idx]}$" ,
                                        plot_error=plot_error)

        if idx == 0:
            ax.set_ylabel("policy\nprobability (HMC)")

        if idx > 0:
            ticks = ax.xaxis.get_major_ticks()
            if len(ticks) > 0:
                ticks[0].label1.set_visible(False)

    sns.despine()

    fig.supxlabel("episode", y=-0.12)

    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('path cumulative\nreward', rotation=270, labelpad=25, x=0.95)

    plt.subplots_adjust(right=0.8, bottom=0.20, wspace=0.25, hspace=0.15)

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()



def plot_deepsea_grid(indices, trajectories, probs, figsize_per_row=(15,5), sim_traj_dict=None, draw_curr_traj=True, draw_background=False, 
                      draw_unexplored=False, action_map=None, pyramid=False, save_path=None, env=None, depth=None, obs_history=None):

    n_plots = len(indices)
    max_cols = 5
    
    n_cols = min(n_plots, max_cols)
    n_rows = math.ceil(n_plots / max_cols)

    if depth is None and env is None:
        assert len(set([len(t) for t in trajectories[1:]])) == 1
        depth = len(trajectories[-1])
    elif depth is None:
        depth = env.depth + 1
    else:
        depth += 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_row[0]*cm, figsize_per_row[1]*n_rows*cm))

    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        
        if i < n_plots:

            frame_idx = indices[i]
            
            render_deepsea(
                ax=ax,
                trajectories=trajectories,
                frame_index=frame_idx,
                depth=depth,
                probs=probs[frame_idx],
                action_map=action_map,
                mode="pie",
                sim_traj_dict=sim_traj_dict, 
                draw_curr_traj=draw_curr_traj, 
                draw_background=draw_background,
                draw_unexplored=draw_unexplored,
                pyramid=pyramid,
                env=env,
                obs_history=obs_history
            )
            ax.set_title(f"ep. {frame_idx}", y=0.9)
        else:
            ax.axis('off')

    if sim_traj_dict is not None and sim_traj_dict["num_sims"] > 0:

        custom_cmap = colors.LinearSegmentedColormap.from_list("alpha_orange", ["white", "#cc5500"])
        
        norm = colors.Normalize(vmin=0, vmax=1)
        sm = mcm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=0.95, bottom=0, top=1)
    
        cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7]) 
        
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([0, 1])  # Only 0 and 1
        cbar.set_label("visitation rate", rotation=270, labelpad=4)
        
    else:
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    
    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_deepsea_init_demo(figsize, env, title=None, pyramid=None, save_path=None):

    depth = env.depth + 1
    
    _, ax = plt.subplots(figsize=(figsize[0]*cm, figsize[1]*cm))

    probs = np.ones(len(env.get_all_states())-len(env.get_all_terminal_states())) * 0.5
    if pyramid is None:
        pyramid = True if isinstance(env, DeepSeaPyramid) or isinstance(env, DeepSeaSwirl) else False

    render_deepsea(
        ax=ax,
        trajectories=[[]],
        frame_index=0,
        depth=depth,
        probs=probs,
        action_map=env.action_map,
        mode="pie",
        sim_traj_dict=None, 
        draw_curr_traj=False, 
        draw_background=True,
        draw_unexplored=False,
        pyramid=pyramid,
        env=env
    )
            
    ax.set_title(title, y=0.9)
        
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    
    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_episode_progress(progress_data_list, method_names, epsilon_list, max_datasize, figsize_per_plot=(3, 5), xtick_freq=None, save_path=None):

    if not isinstance(progress_data_list, list):
        progress_data_list = [progress_data_list]
    
    if isinstance(method_names, str):
        method_names = [method_names]
        
    num_methods = len(progress_data_list)
    num_epsilons = len(epsilon_list)

    fig, axes = plt.subplots(1, num_methods, figsize=(figsize_per_plot[0]*num_methods*cm, figsize_per_plot[1]*cm), sharex=True, sharey=True)
    
    if num_methods == 1:
        axes = [axes]

    colors_list = plt.cm.viridis(np.linspace(0, 0.9, num_epsilons))

    for method_idx, method_data in enumerate(progress_data_list):
        ax = axes[method_idx]
        current_method_name = method_names[method_idx]
        
        for eps_idx, eps in enumerate(epsilon_list):
            current_color = colors_list[eps_idx]

            data = method_data[eps_idx]

            mean = np.mean(data, axis=0) / max_datasize
            std = np.std(data, axis=0) / max_datasize
            episodes = np.arange(len(mean))
            
            ax.plot(episodes, mean, 
                    label=rf"$\epsilon={eps}$", 
                    color=current_color, 
                    linestyle="-", 
                    linewidth=1)

            ax.fill_between(episodes, mean - std, mean + std, 
                            color=current_color, 
                            alpha=0.15)

        ax.set_title(current_method_name)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        max_len = max([len(m[0]) if len(m.shape)>1 else len(m) for m in method_data])
        ax.set_xlim(0, max_len)
        ax.set_ylim(-0.05, 1.05)

        if xtick_freq is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_freq))

        if method_idx > 0:
            plt.setp(ax.get_xticklabels()[0], visible=False)

        if method_idx == 0:
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False, left=False)

    fig.supylabel(rf"\% explored", x=0.04, y=0.55)
    fig.supxlabel("episode", y=0.)
    
    sns.despine()
    epsilon_handles = [Line2D([0], [0], color=colors_list[i], lw=2, label=rf'$\epsilon={eps}$') 
                       for i, eps in enumerate(epsilon_list)]
    
    fig.legend(
        handles=epsilon_handles,
        loc='center left',
        bbox_to_anchor=(0.91, 0.55),
        frameon=False,
        title="tolerance"
    )

    plt.subplots_adjust(bottom=0.30, wspace=0.1)

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()



def plot_pyramid_rank(rank_data_list, method_names, ours_labels, epsilon_list, figsize_per_plot=(3, 5), xtick_freq=None, save_path=None):

    if not isinstance(rank_data_list, list):
        rank_data_list = [rank_data_list]
    
    if isinstance(method_names, str):
        method_names = [method_names]
        
    num_methods = len(rank_data_list)
    num_epsilons = len(epsilon_list)

    global_max_y = 0
    max_len = 0
    for method_data in rank_data_list:
        for eps_idx in range(num_epsilons):
            data = method_data[eps_idx]
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            global_max_y = max(global_max_y, np.max(mean + std))
            max_len = max(max_len, len(mean))

    fig, axes = plt.subplots(1, num_epsilons, figsize=(figsize_per_plot[0]*num_epsilons*cm, figsize_per_plot[1]*cm), sharex=True, sharey=True)
    
    if num_methods == 1:
        axes = [axes]

    colors_list = sns.color_palette("colorblind")

    for eps_idx, eps in enumerate(epsilon_list):
        ax = axes[eps_idx]

        for method_idx, method_data in enumerate(rank_data_list):
            current_color = colors_list[method_idx % len(colors_list)]

            data = method_data[eps_idx]

            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            xrange = np.arange(len(mean)) + 1

            ax.plot(xrange, mean, 
                    label=rf"$\epsilon={eps}$", 
                    color=current_color, 
                    linestyle="-", 
                    linewidth=1)

            ax.fill_between(xrange, mean - std, mean + std, 
                            color=current_color,
                            alpha=0.15)

        ax.plot(xrange, np.ones_like(xrange)/len(xrange), color='gray', linestyle='--', linewidth=1.2, alpha=0.8, zorder=10)

        ax.set_title(rf"$\epsilon={eps}$")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.set_xlim(1, max_len)
        ax.set_ylim(-0.02, global_max_y * 1.05)

        freq = xtick_freq if xtick_freq is not None else 2
        ax.set_xticks(np.arange(1, max_len, freq))

        if eps_idx == 0:
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False, left=False)

    fig.supylabel("probability", x=0.04, y=0.6)
    fig.supxlabel("rank", y=0.15)
    
    sns.despine()

    method_handles = []
    for idx in range(len(method_names)):
        ours_label = " (ours)" if isinstance(ours_labels, list) and idx in ours_labels else ""
        method_handles.append(Line2D([0], [0], color=colors_list[idx % len(colors_list)], lw=1.5, label=method_names[idx] + ours_label))


    fig.legend(
        handles=method_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(method_handles),
        frameon=False,
        handlelength=1.0,
    )


    plt.subplots_adjust(bottom=0.35, wspace=0.1)

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_tvd_errors(epsilon_list, sampling_err, approx_err, figsize=(10, 8), save_path=None):

    _, ax = plt.subplots(figsize=(figsize[0]*cm, figsize[1]*cm))
    
    samp_mean = np.array([x[0] for x in sampling_err])
    samp_std = np.array([x[1] for x in sampling_err])
    
    app_mean = np.array([x[0] for x in approx_err])
    app_std = np.array([x[1] for x in approx_err])
    
    tot_mean = samp_mean + app_mean
    tot_std = np.sqrt(samp_std**2 + app_std**2)
    
    colors_list = sns.color_palette("colorblind", 3)
    labels = ["sampling error", "approximation error", "sum of errors"]
    means = [samp_mean, app_mean, tot_mean]
    stds = [samp_std, app_std, tot_std]
    
    for i in range(3):
        ax.plot(epsilon_list, means[i], label=labels[i], 
                color=colors_list[i], linewidth=1.5)
        
        ax.fill_between(epsilon_list, 
                        means[i] - stds[i], 
                        means[i] + stds[i], 
                        color=colors_list[i], alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel(r"$\epsilon$ (tolerance)")
    ax.set_ylabel("average TVD")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.set_xticks(epsilon_list)
    
    sns.despine()

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        handlelength=1.5
    )

    plt.subplots_adjust(right=0.9)        
    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
            
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def load_config(config_dir, config_name):
    config_path = os.path.join(config_dir, config_name)
    omega_config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(omega_config, resolve=True)
    return config


def retrieve_filedirs(config, data_dir, epsilon_list, stepsize_list=None):
    paths = []
    config = config.copy()
    method = config["method"]
    for idx, epsilon in enumerate(epsilon_list):
        config["epsilon"] = epsilon
        if method == "hmc":
            config["step_size"] = stepsize_list[idx]
        paths.append(find_existing_run(base_dir=data_dir, config=config))

    return paths


def retrieve_data_regret_plot(config, paths, epsilon_list, num_repeats, max_reward=None):
    optimal_total_reward = 1 - config["depth"] * config.get("penalty", 0) if max_reward is None else max_reward
    num_episodes = config["num_episodes"]
    
    regret_plot = []
    
    for idx, _ in enumerate(epsilon_list): # contains different epislon
        regret_episodes = np.zeros((num_repeats, num_episodes))
        
        for rep in range(num_repeats):
            path = os.path.join(paths[idx], f"run_{rep}.npy")
            reward_history = np.load(path, allow_pickle=True).item()["reward_history"][1:]
            for episode in range(len(reward_history)):
                total_reward = np.array(reward_history[episode]).sum()
                regret_episodes[rep, episode] = optimal_total_reward - total_reward
    
        regret_plot.append(regret_episodes)
    return regret_plot


def retrieve_hmc_accprob(config, paths, epsilon_list, num_repeats):
    num_episodes = config["num_episodes"]
    
    acc_list = []
    
    for idx, _ in enumerate(epsilon_list): # contains different epislon
        acc_episodes = np.zeros((num_repeats, num_episodes))
        
        for rep in range(num_repeats):
            path = os.path.join(paths[idx], f"run_{rep}.npy")
            acc_history = np.load(path, allow_pickle=True).item()["acc_prob_history"][1:]
            acc_episodes[rep] = acc_history
    
        acc_list.append(acc_episodes)
    return acc_list


def make_virtual_env(config, action_map=None, make_det=True):
    use_penalty = True
    env_name = config["name"]
    if env_name.lower() == "deepsea":
        env_class = DeepSea
    elif env_name.lower() == "deepseapyramid":
        env_class = DeepSeaPyramid
        use_penalty = False
    elif env_name.lower() == "deepseaswirl":
        env_class = DeepSeaSwirl
    else:
        raise ValueError(f"undefined env_name {env_name}.")

    if make_det:
        env_kwargs = {"depth": config["depth"], "deterministic_transition": True, "randomised_actions": False}
    else:
        env_kwargs = {k: config[k] for k in ["depth", "deterministic_transition", "sto_trans_prob", "randomised_actions"]}
    if use_penalty:
        env_kwargs["penalty"] = config["penalty"]
    env_kwargs["action_map"] = action_map

    env = env_class(**env_kwargs)
    return env


def find_policy_reward(config, env, paths):

    optimal_paths_rewards = np.zeros(len(paths))

    if config["deterministic_transition"]:
        for path_idx in range(len(paths)):
            reward = 0
            for state, action in paths[path_idx]:
                reward += env.reward(state=state, action=action, next_state=None)
            optimal_paths_rewards[path_idx] = reward

    else:
        raise NotImplementedError()

    return optimal_paths_rewards


def det_standardised_path_to_id(path):
    output_id = ""
    for _, action in path:
        output_id += str(action)
    return output_id


def standardise_path(path, action_map, reference_action_map):
    standardised_path = []
    for state, action in path:
        if action_map[*state] == reference_action_map[*state]:
            standardised_action = action
        else:
            standardised_action = 1 - action
        standardised_path.append((state, standardised_action))
    return standardised_path


def estimate_optimal_path_prob_hmc(samples, env_det, action_map, reference_action_map, standardised_paths):
    burnin = len(samples) // 2
    optimal_prob = np.zeros(len(standardised_paths))
    for path_idx, path in enumerate(standardised_paths):
        bool_tmp = np.ones(len(samples)-burnin, dtype=np.bool)
        for state, std_action in path:
            if action_map[*state] == reference_action_map[*state]:
                action = std_action
            else:
                action = 1 - std_action
            bool_tmp = bool_tmp & (samples[burnin:, env_det.nu(state=state, action=action)-1] > samples[burnin:, env_det.nu(state=state, action=1-action)-1])
        optimal_prob[path_idx] = np.mean(bool_tmp)
    return optimal_prob


def get_standardised_env_and_paths(config):
    env_det = make_virtual_env(config=config, action_map=None)
    standardised_all_paths, _ = get_all_deterministic_paths(env_det)
    return env_det, standardised_all_paths


def retrieve_path_probs(config, directory, num_repeats, env_det, standardised_all_paths):

    optimal_probs_array = None

    if config["deterministic_transition"] is not True:
        raise NotImplementedError()

    # make history_id -> position_id
    standardised_action_map = env_det.action_map
    path_id_to_pos_dict = {det_standardised_path_to_id(path=ph): i for i, ph in enumerate(standardised_all_paths)}

    for rep in range(num_repeats):
        path_rep = os.path.join(directory, f"run_{rep}.npy")
        data = np.load(path_rep, allow_pickle=True).item()
        if config["method"].lower()[:3] == "cdf":
            all_paths, all_optimal_path_probs, _ = data["paths_data"]
            _, action_map = data["probs_data"]
    
            if rep == 0: # initialise array
                optimal_probs_array = np.zeros((num_repeats, len(all_optimal_path_probs), len(all_paths)))
            
            for episode in range(len(all_optimal_path_probs)):
                for path_idx, optimal_prob in enumerate(all_optimal_path_probs[episode]):
                    standardised_path = standardise_path(path=all_paths[path_idx], action_map=action_map, reference_action_map=standardised_action_map)
                    idx = path_id_to_pos_dict[det_standardised_path_to_id(path=standardised_path)]
                    optimal_probs_array[rep, episode, idx] = optimal_prob
        elif config["method"].lower() == "hmc":
            samples_history = data["samples_history"]
            action_map, _ = data["env_data"]

            if rep == 0:
                optimal_probs_array = np.zeros((num_repeats, len(samples_history), len(standardised_all_paths)))

            for episode in range(len(samples_history)):
                optimal_probs_array[rep, episode] = estimate_optimal_path_prob_hmc(samples=samples_history[episode], env_det=env_det, action_map=action_map, 
                                               reference_action_map=standardised_action_map, standardised_paths=standardised_all_paths)

    return optimal_probs_array


def retrieve_data_demo_plot(config, directory, to_plot_run=0, return_obs=False):

    data = np.load(os.path.join(directory, f"run_{to_plot_run}.npy"), allow_pickle=True).item()
    
    state_history = data["state_history"]
    all_probs, action_map = data["probs_data"]
    all_paths, all_optimal_path_probs, _ = data["paths_data"]

    env = make_virtual_env(config, action_map=action_map, make_det=False)
    if not return_obs:
        return all_paths, all_optimal_path_probs, env, state_history, all_probs, action_map
    else:
        obs_history = data["obs_history"]
        return all_paths, all_optimal_path_probs, env, state_history, all_probs, action_map, obs_history


def retrieve_episode_progress(num_repeats, epsilon_list, paths):
    output = []

    for i in range(len(epsilon_list)):
        completion_time_array = []
        for j in range(num_repeats):
            data = np.load(os.path.join(paths[i], f"run_{j}.npy"), allow_pickle=True).item()
            obs_history = data["obs_history"]
            completion_time_array.append(np.array([len(obs["state0"]) for obs in obs_history]))

        output.append(np.array(completion_time_array))

    return output


def retrieve_pyramid_rank(num_repeats, epsilon_list, config, paths, episode, env_det, standardised_all_paths, optimal_paths_rewards):
    output = []

    optimal_paths_indices = np.arange(len(optimal_paths_rewards))[optimal_paths_rewards == 1]
    for idx in range(len(epsilon_list)):
        data = retrieve_path_probs(config=config, directory=paths[idx], num_repeats=num_repeats, env_det=env_det, standardised_all_paths=standardised_all_paths)
        output.append(np.sort(data[:, episode, optimal_paths_indices], axis=-1)[..., ::-1])

    return output


def average_tvd(p, q):
    abs_diff = np.abs(p - q)
    return 0.5 * np.sum(abs_diff, axis=-1).mean(), 0.5 * np.sum(abs_diff, axis=-1).mean(-1).std()
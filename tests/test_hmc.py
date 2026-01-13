import pytest
import numpy as np
import itertools
from ABRLExact.environment import DeepSea
from ABRLExact.HMCAgent import run_hmc_deepsea
from ABRLExact.utils import compute_q_optimal_probs, add_obs

def test_hmc_deepsea_guided():
    epsilon = 0.1
    sigma = 10

    env = DeepSea(depth=1, starting_state=(0, 0), goal_state=(-1, -1), penalty=0.2, randomised_actions=False)
    
    obs = {"state0": [], "action": [], "state1": [], "rewards": [], "done": []}

    for state in env.get_all_states():
        if state not in env.get_all_terminal_states():
            for action in range(2):
                next_state, reward, done = env.step(action=action, is_simulation=True, simulation_state=state)
                add_obs(obs=obs, unique_stat=set(), state0=state, action=action, state1=next_state, reward=reward, done=done)
    
    hmc_state_history, hmc_reward_history, samples_history = run_hmc_deepsea(env, epsilon=epsilon, sigma=sigma, num_episodes=10, num_samples=1000, 
                                                                         num_warmup_runs=2, num_warmup_samples_per_run=10, target_acc_prob=0.75, 
                                                                         step_size_rates=(1.3, 0.7, 1.1, 0.8), step_size=0.1, num_steps=10, 
                                                                         disable_progbar=False, input_obs=[obs], output_obs=False)

    all_action_combinations = list(itertools.product([0, 1], repeat=env.depth))
    all_paths = []
    for comb in all_action_combinations:
        path = []
        curr_state = (0, 0)
        for action in comb:
            next_state, _, _ = env.step(action, is_simulation=True, simulation_state=curr_state)
            path.append((curr_state, action))
            curr_state = next_state
        all_paths.append(path)
    
    test_hmc = np.zeros((len(samples_history), len(all_paths)))
    for episode, samples in enumerate(samples_history):
        path_probs = compute_q_optimal_probs(samples, all_paths, env)
        test_hmc[episode] = path_probs

    posterior_means = samples_history[-1].mean(0)
    
    assert posterior_means[0] == pytest.approx(0.8, abs=0.05)
    assert posterior_means[1] == pytest.approx(0.2, abs=0.05)

    assert test_hmc[-1][0] == pytest.approx(1.0, abs=0.04)
    assert test_hmc[-1][1] == pytest.approx(0.0, abs=0.04)
import pytest
import numpy as np
from ABRLExact.environment import fourDMDP, DeepSea
from ABRLExact.CDFAgent import cdf_solution, run_cdf_deepsea

def test_cdf_solution_fourDMDP():
    env = fourDMDP()
    obs = dict()
    obs["state0"] = [1, 1, 2, 2, 3]
    obs["action"] = [1, 2, 3, 4, 5]
    obs["state1"] = []
    obs["rewards"] = []
    
    for i in range(len(obs["state0"])):
        next_state, reward, _ = env.step(obs["action"][i], is_simulation=True, simulation_state=obs["state0"][i])
        obs["state1"].append(next_state)
        obs["rewards"].append(reward)

    epsilon = 0.01
    sigma = 2.0

    # Test Case 1
    state_q_list1 = [1, 2, 3]
    action_q_list1 = [1, 3, 5]
    result1 = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list1, action_q_list=action_q_list1)
    
    # Expected: 0.5057769735077257
    assert result1 == pytest.approx(0.5057769735077257, rel=1e-6)

    # Test Case 2
    state_q_list2 = [1, 2]
    action_q_list2 = [1, 4]
    result2 = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list2, action_q_list=action_q_list2)
    
    # Expected: 0.4942230264922744
    assert result2 == pytest.approx(0.4942230264922744, rel=1e-6)

    # Test Case 3
    state_q_list3 = [1, 3]
    action_q_list3 = [2, 5]
    result3 = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list3, action_q_list=action_q_list3)
    
    # Expected: 0.0
    assert result3 == pytest.approx(0.0, abs=1e-6)

def test_cdf_solution_fourDMDP_independence():
    env = fourDMDP()
    obs = dict()
    obs["state0"] = [1,2,3]
    obs["action"] = [1,3,5]
    obs["state1"] = []
    obs["rewards"] = []
    for i in range(len(obs["state0"])):
        next_state, reward, _ = env.step(obs["action"][i], is_simulation=True, simulation_state=obs["state0"][i])
        obs["state1"].append(next_state)
        obs["rewards"].append(reward)

    epsilon = 0.01
    sigma = 5.

    state_q_list1 = [1,2]
    action_q_list1 = [1,4]

    # Test Case 1
    result1 = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list1, action_q_list=action_q_list1)
    assert result1 == pytest.approx(0.35020252822343373, rel=1e-6)

    # Test Case 2
    result2 = cdf_solution(obs, env, epsilon, sigma, state_q=state_q_list1[0])[0][0] * cdf_solution(obs, env, epsilon, sigma, state_q=state_q_list1[1])[0][1]
    assert result2 == pytest.approx(0.28567728040579937, rel=1e-6)

    # Test Case 3
    result3 = cdf_solution(obs, env, epsilon, sigma, state_q=state_q_list1[0], normalise=False)[0][0] * cdf_solution(obs, env, epsilon, sigma, state_q=state_q_list1[1], normalise=False)[0][1]
    assert result3 == pytest.approx(0.28567728040579937, rel=1e-6)

def test_cdf_solution_fourDMDP_single_action():
    env = fourDMDP()
    obs = dict()
    obs["state0"] = [1,2]
    obs["action"] = [2,4]
    obs["state1"] = []
    obs["rewards"] = []
    for i in range(len(obs["state0"])):
        next_state, reward, _ = env.step(obs["action"][i], is_simulation=True, simulation_state=obs["state0"][i])
        obs["state1"].append(next_state)
        obs["rewards"].append(reward)
    
    epsilon = 0.01
    sigma = 5.0

    result = cdf_solution(obs, env, epsilon, sigma, state_q=3)[0]
    assert result == pytest.approx(1.0, abs=1e-6)

def test_cdf_solution_fourDMDP_state_q_query():
    env = fourDMDP()
    obs = dict()
    obs["state0"] = [1,1,2,2,3]
    obs["action"] = [1,2,3,4,5]
    obs["state1"] = []
    obs["rewards"] = []
    for i in range(len(obs["state0"])):
        next_state, reward, _ = env.step(obs["action"][i], is_simulation=True, simulation_state=obs["state0"][i])
        obs["state1"].append(next_state)
        obs["rewards"].append(reward)

    epsilon = 0.01
    sigma = 2.
    state_q = 1
    
    # Case 1: Normalise=True
    result1 = cdf_solution(obs, env, epsilon, sigma, state_q=state_q, state_q_list=None, action_q_list=None)
    probs1, actions1 = result1
    
    assert actions1 == [1, 2]
    assert probs1[0] == pytest.approx(1.0, abs=1e-6)
    assert probs1[1] == pytest.approx(0.0, abs=1e-6)

    # Case 2: Normalise=False
    result2 = cdf_solution(obs, env, epsilon, sigma, state_q=state_q, state_q_list=None, action_q_list=None, normalise=False)
    probs2, actions2 = result2
    
    assert actions2 == [1, 2]
    assert probs2[0] == pytest.approx(1.0, abs=1e-6)
    assert probs2[1] == pytest.approx(0.0, abs=1e-6)

def test_cdf_solution_fourDMDP_repeated_obs():
    env = fourDMDP()
    obs = dict()
    obs["state0"] = [1,1,2,2,3] * 10
    obs["action"] = [1,2,3,4,5] * 10
    obs["state1"] = []
    obs["rewards"] = []
    for i in range(len(obs["state0"])):
        next_state, reward, _ = env.step(obs["action"][i], is_simulation=True, simulation_state=obs["state0"][i])
        obs["state1"].append(next_state)
        obs["rewards"].append(reward)

    epsilon = 0.01
    sigma = 2.
    
    # Test Case 1
    state_q_list1 = [1,2,3]
    action_q_list1 = [1,3,5]
    result1 = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list1, action_q_list=action_q_list1)
    assert result1 == pytest.approx(0.5018228465160195, rel=1e-6)

    # Test Case 2
    state_q_list2 = [1,2]
    action_q_list2 = [1,4]
    result2 = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list2, action_q_list=action_q_list2)
    assert result2 == pytest.approx(0.49817715348398034, rel=1e-6)

    # Test Case 3
    state_q_list3 = [1,3]
    action_q_list3 = [2,5]
    result3 = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list3, action_q_list=action_q_list3)
    assert result3 == pytest.approx(0.0, abs=1e-6)

def test_run_cdf_deepsea():
    env = DeepSea(depth=3, 
            starting_state=(0,0), 
            goal_state=(-1,-1), 
            deterministic_transition=True, 
            randomised_actions=True, 
            randomised_action_seed=None, 
            penalty=0.02,
            sto_trans_prob=None)

    state_history, (all_probs, action_map), (all_paths, all_optimal_path_probs, deterministic_transition), reward_history, obs_history = run_cdf_deepsea(  # pylint: disable=unused-variable
        env, epsilon=0.02, sigma=10, num_episodes=30, use_qmc=True, qmc_sobol_power=14, output_obs=True, save_path=None, disable_tqdm=True)

    idx = np.argmax(all_optimal_path_probs[-1]).item()
    assert np.all([(all_paths[idx][i][0] == (i,i)) & (all_paths[idx][i][1] != action_map[i,i].item()) for i in range(len(all_paths[idx]))])

def test_cdf_solution_caching_deepsea():
    env = DeepSea(depth=3, 
            starting_state=(0,0), 
            goal_state=(-1,-1), 
            deterministic_transition=True, 
            randomised_actions=True, 
            randomised_action_seed=42, 
            penalty=0.02,
            sto_trans_prob=None)


    obs = {"state0": [], "action": [], "state1": [], "rewards": [], "done": []}
    curr_state = (0,0)

    actions = [0, 1] 
    for action in actions:
        next_state, reward, done = env.step(action, is_simulation=True, simulation_state=curr_state)
        obs["state0"].append(curr_state)
        obs["action"].append(action)
        obs["state1"].append(next_state)
        obs["rewards"].append(reward)
        obs["done"].append(done)
        curr_state = next_state

    epsilon = 0.02
    sigma = 10.0
    
    state_q_list1 = [(0,0), (1,0)]
    action_q_list1 = [0, 0]
    
    cache = {}
    prob1 = cdf_solution(obs, env, epsilon, sigma, state_q=None, 
                                    state_q_list=state_q_list1, action_q_list=action_q_list1, 
                                    cache=cache)
    
    assert "pos_mean_array" in cache
    assert "pos_cov_array" in cache
    assert "pr_vec" in cache
    
    prob1_cached = cdf_solution(obs, env, epsilon, sigma, state_q=None, 
                                   state_q_list=state_q_list1, action_q_list=action_q_list1, 
                                   cache=cache)
    
    assert prob1 == pytest.approx(prob1_cached, abs=1e-5)
    
    state_q_list2 = [(0,0), (1,1)]
    action_q_list2 = [1, 1]
    
    prob2_cached = cdf_solution(obs, env, epsilon, sigma, state_q=None, 
                                   state_q_list=state_q_list2, action_q_list=action_q_list2, 
                                   cache=cache)
    
    prob2_uncached = cdf_solution(obs, env, epsilon, sigma, state_q=None, 
                                  state_q_list=state_q_list2, action_q_list=action_q_list2)
    
    assert prob2_cached == pytest.approx(prob2_uncached, abs=1e-5)

def test_cdf_solution_caching_state_q_change_deepsea():
    """
    Test that caching works when the query 'state_q' changes (Mode 0), 
    verifying that the cached posterior is valid for different queries.
    """
    env = DeepSea(depth=3, 
            starting_state=(0,0), 
            goal_state=(-1,-1), 
            deterministic_transition=True, 
            randomised_actions=True, 
            randomised_action_seed=42, 
            penalty=0.02,
            sto_trans_prob=None)

    obs = {"state0": [], "action": [], "state1": [], "rewards": [], "done": []}
    curr_state = (0,0)
    action = 0
    next_state, reward, done = env.step(action, is_simulation=True, simulation_state=curr_state)
    obs["state0"].append(curr_state)
    obs["action"].append(action)
    obs["state1"].append(next_state)
    obs["rewards"].append(reward)
    obs["done"].append(done)

    epsilon = 0.02
    sigma = 10.0
    
    cache = {}
    _, _ = cdf_solution(obs, env, epsilon, sigma, state_q=(0,0), cache=cache)
    
    probs_cached, _ = cdf_solution(obs, env, epsilon, sigma, state_q=(1,0), cache=cache)
    
    probs_uncached, _ = cdf_solution(obs, env, epsilon, sigma, state_q=(1,0))
    
    assert np.allclose(probs_cached, probs_uncached, atol=1e-5)
import pytest
from ABRLExact.environment import fourDMDP
from ABRLExact.CDFAgent import exact_solution

def test_exact_solution_fourDMDP():
    # Setup environment and observations
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
    result1 = exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list1, action_q_list=action_q_list1)
    
    # Expected: 0.5057769735077257
    assert result1 == pytest.approx(0.5057769735077257, rel=1e-6)

    # Test Case 2
    state_q_list2 = [1, 2]
    action_q_list2 = [1, 4]
    result2 = exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list2, action_q_list=action_q_list2)
    
    # Expected: 0.4942230264922744
    assert result2 == pytest.approx(0.4942230264922744, rel=1e-6)

    # Test Case 3
    state_q_list3 = [1, 3]
    action_q_list3 = [2, 5]
    result3 = exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list3, action_q_list=action_q_list3)
    
    # Expected: 0.0
    assert result3 == pytest.approx(0.0, abs=1e-6)

def test_exact_solution_fourDMDP_independence():
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
    result1 = exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list1, action_q_list=action_q_list1)
    assert result1 == pytest.approx(0.35020252822343373, rel=1e-6)

    # Test Case 2
    result2 = exact_solution(obs, env, epsilon, sigma, state_q=state_q_list1[0])[0][0] * exact_solution(obs, env, epsilon, sigma, state_q=state_q_list1[1])[0][1]
    assert result2 == pytest.approx(0.28567728040579937, rel=1e-6)

    # Test Case 3
    result3 = exact_solution(obs, env, epsilon, sigma, state_q=state_q_list1[0], normalise=False)[0][0] * exact_solution(obs, env, epsilon, sigma, state_q=state_q_list1[1], normalise=False)[0][1]
    assert result3 == pytest.approx(0.28567728040579937, rel=1e-6)

def test_exact_solution_fourDMDP_single_action():
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

    result = exact_solution(obs, env, epsilon, sigma, state_q=3)[0]
    assert result == pytest.approx(1.0, abs=1e-6)

def test_exact_solution_fourDMDP_state_q_query():
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
    result1 = exact_solution(obs, env, epsilon, sigma, state_q=state_q, state_q_list=None, action_q_list=None)
    probs1, actions1 = result1
    
    assert actions1 == [1, 2]
    assert probs1[0] == pytest.approx(1.0, abs=1e-6)
    assert probs1[1] == pytest.approx(0.0, abs=1e-6)

    # Case 2: Normalise=False
    result2 = exact_solution(obs, env, epsilon, sigma, state_q=state_q, state_q_list=None, action_q_list=None, normalise=False)
    probs2, actions2 = result2
    
    assert actions2 == [1, 2]
    assert probs2[0] == pytest.approx(1.0, abs=1e-6)
    assert probs2[1] == pytest.approx(0.0, abs=1e-6)

def test_exact_solution_fourDMDP_repeated_obs():
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
    result1 = exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list1, action_q_list=action_q_list1)
    assert result1 == pytest.approx(0.5018228465160195, rel=1e-6)

    # Test Case 2
    state_q_list2 = [1,2]
    action_q_list2 = [1,4]
    result2 = exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list2, action_q_list=action_q_list2)
    assert result2 == pytest.approx(0.49817715348398034, rel=1e-6)

    # Test Case 3
    state_q_list3 = [1,3]
    action_q_list3 = [2,5]
    result3 = exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list3, action_q_list=action_q_list3)
    assert result3 == pytest.approx(0.0, abs=1e-6)
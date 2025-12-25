from copy import deepcopy

import numpy as np
from scipy.stats import multivariate_normal

def exact_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=None, action_q_list=None):
    '''
    obs: dictionary of state0, action, state1, rewards
    env: initialised MDP class
    epsilon: likelihood variance
    sigma: prior variance
    state_q: If query is for the probability of optimality of actions of a given state_q, set state_q
    state_q_list: If query is for the probability of optimality of a sequence of state action pairs, set state_q_list and action_q_list instead, and keep state_q as None
    action_q_list: same as state_q_list
    '''

    if state_q is not None and action_q_list is None:
        mode = 0
    else:
        assert state_q is None and state_q_list is not None and action_q_list is not None
        mode = 1
    
    n = len(obs["state0"])
    
    states = env.get_all_states()
    goal_states = env.get_all_terminal_states()

    theta_len = 0
    permissible_actions_dict = dict()
    for idx, state in enumerate(states):
        actions = env.get_possible_actions(state, gym_space=False)
        permissible_actions_dict[str(state)] = actions
        if state not in goal_states:
            theta_len += len(actions)

    if mode == 0:
        action_q_list = permissible_actions_dict[str(state_q)]
    
    unique_state1 = list(set(obs["state1"]))
    is_ell_list_empty = np.all([state in goal_states for state in unique_state1])

    if is_ell_list_empty:
        AprimeD_states = []
        AprimeD = []
        ell_list = [[None]]
    else:
        AprimeD_states, AprimeD = list(zip(*[[state, permissible_actions_dict[str(state)]] for state in unique_state1 if state not in goal_states]))
        ell_list = np.array(np.meshgrid(*AprimeD)).T.reshape(-1,len(AprimeD_states))
    
    unique_state1_nongoal = AprimeD_states

    #Get B_ell
    B_ell_list = []
    for ell in ell_list:
        B_ell = np.zeros([n,theta_len])
        for i in range(n):
            for j in range(theta_len):
                if j == (env.nu(state=obs["state0"][i], action=obs["action"][i])-1):
                    B_ell[i,j] += 1
                state1 = obs["state1"][i]
                if state1 not in goal_states:
                    ell_idx = unique_state1_nongoal.index(state1)
                    if j == (env.nu(state=state1, action=ell[ell_idx])-1):
                        B_ell[i,j] -= 1

        B_ell_list.append(B_ell)

    #Get p(r)
    pr_vec = np.zeros(len(ell_list))
    if len(obs["rewards"]) > 0:
        Gamma_inv_list = []
        for idx in range(len(ell_list)):
            B_ell = B_ell_list[idx]
            Gamma_inv_list.append(sigma**2 * B_ell @ B_ell.T + epsilon**2 * np.eye(n))
            pr_vec[idx] = multivariate_normal.pdf(obs["rewards"],mean=np.zeros(n),cov=Gamma_inv_list[idx])
    
        #Get p(theta|r)
        pos_mean_list = []
        pos_cov_list = []
        
        for idx in range(len(ell_list)):
            B_ell = B_ell_list[idx]
            Gamma_inv = Gamma_inv_list[idx]
    
            pos_cov_ = B_ell.T @ np.linalg.inv(Gamma_inv)
            
            pos_mean = np.matmul(sigma**2 * pos_cov_, obs["rewards"])
            pos_mean_list.append(pos_mean)
            
            pos_cov = sigma**2 * np.eye(theta_len) - sigma**4 * pos_cov_  @ B_ell
            pos_cov_list.append(pos_cov)
    else:
        pos_mean_list = [np.zeros(theta_len)]
        pos_cov_list = [sigma**2 * np.eye(theta_len)]
        pr_vec[0] = 1.
    

    #Get Eell transformation matrix
    D_ell_list = []
    if mode == 0:
        D_ell_star_list = [[] for i in range(len(action_q_list))]
    elif mode == 1:
        D_ell_star_list = []

    for ell in ell_list:
        D_ell = []
        D_ell_star = []
        for state_idx, state1 in enumerate(unique_state1_nongoal):
            for action in permissible_actions_dict[str(state1)]:
                action_ell = ell[state_idx]
                if not (action == action_ell):
                    D_ell_tmp =  np.zeros(theta_len)
                    D_ell_tmp[env.nu(state=state1, action=action)-1] = 1
                    D_ell_tmp[env.nu(state=state1, action=action_ell)-1] = -1
                    D_ell.append(D_ell_tmp)

        if mode == 0:
            for action_idx, action_q in enumerate(action_q_list):
                D_ell_star = deepcopy(D_ell)
                if state_q not in unique_state1_nongoal:
                    for action in permissible_actions_dict[str(state_q)]:
                        if not (action == action_q):
                            D_ell_star_tmp = np.zeros(theta_len)
                            D_ell_star_tmp[env.nu(state=state_q, action=action)-1] = 1
                            D_ell_star_tmp[env.nu(state=state_q, action=action_q)-1] = -1
                            D_ell_star.append(D_ell_star_tmp)
    
                else:
                    state_q_idx = unique_state1_nongoal.index(state_q)
                    if ell[state_q_idx] == action_q: #E* and Ell align, set D_ell_star as D_ell
                        pass
                    else: #otherwise, set it as None, to be dealt with later
                        D_ell_star = None
                
                D_ell_star = np.array(D_ell_star)
                D_ell_star_list[action_idx].append(D_ell_star)

        elif mode == 1:
            D_ell_star = deepcopy(D_ell)
            for state_q_idx, state_q in enumerate(state_q_list):
                action_q = action_q_list[state_q_idx]
                if state_q not in unique_state1_nongoal:
                    for action in permissible_actions_dict[str(state_q)]:
                        if not (action == action_q):
                            D_ell_star_tmp = np.zeros(theta_len)
                            D_ell_star_tmp[env.nu(state=state_q, action=action)-1] = 1
                            D_ell_star_tmp[env.nu(state=state_q, action=action_q)-1] = -1
                            D_ell_star.append(D_ell_star_tmp)
    
                else:
                    state_q_idx = unique_state1_nongoal.index(state_q)
                    if ell[state_q_idx] == action_q:
                        pass
                    else:
                        D_ell_star = None
                        break
    
            D_ell_star = np.array(D_ell_star)
            D_ell_star_list.append(D_ell_star)


        D_ell = np.array(D_ell)
        D_ell_list.append(D_ell)

    
    
    # Get pEr and pEsr
    pEr_vec = np.zeros(len(ell_list))
    if mode == 0:
        pEstarr_list = [np.zeros(len(ell_list)) for i in range(len(action_q_list))]
    elif mode == 1:
        pEstarr_vec = np.zeros(len(ell_list))
    
    for idx in range(len(ell_list)):
        D_ell = D_ell_list[idx]
        if len(D_ell) == 0:
            pEr_vec[idx] = 1
        else:
            pEr_mean = D_ell @ pos_mean_list[idx]
            pEr_cov = D_ell @ pos_cov_list[idx] @ D_ell.T
            pEr_vec[idx] = multivariate_normal.cdf(x=np.zeros(len(D_ell)), mean=pEr_mean, cov=pEr_cov)

        if mode == 0:
            for action_idx in range(len(action_q_list)):
                D_ell_star = D_ell_star_list[action_idx][idx]
                if not (D_ell_star.shape == ()): #if it is not np.array(None)
                    pEstarr_mean = D_ell_star @ pos_mean_list[idx]
                    pEstarr_cov = D_ell_star @ pos_cov_list[idx] @ D_ell_star.T
                    pEstarr_list[action_idx][idx] = multivariate_normal.cdf(x=np.zeros(len(D_ell_star)), mean=pEstarr_mean, cov=pEstarr_cov)
                else: #if it is np.array(None), prob is 0
                    pEstarr_list[action_idx][idx] = 0.
        elif mode == 1:
            D_ell_star = D_ell_star_list[idx]
            if not (D_ell_star.shape == ()): #if it is not np.array(None)
                pEstarr_mean = D_ell_star @ pos_mean_list[idx]
                pEstarr_cov = D_ell_star @ pos_cov_list[idx] @ D_ell_star.T
                pEstarr_vec[idx] = multivariate_normal.cdf(x=np.zeros(len(D_ell_star)), mean=pEstarr_mean, cov=pEstarr_cov)
            else: #if it is np.array(None), prob is 0
                pEstarr_vec[idx] = 0.

    # Get final answer
    pEstarr_denom = np.sum(pr_vec * pEr_vec)

    if mode == 0:
        pEstarr_numer = np.array([np.sum(pr_vec * pEstarr_vec) for pEstarr_vec in pEstarr_list])
        
        pEstarr = pEstarr_numer / pEstarr_denom
        return pEstarr, action_q_list

    elif mode == 1:
        pEstarr_numer = np.sum(pr_vec * pEstarr_vec)
        
        pEstarr = pEstarr_numer / pEstarr_denom
        return pEstarr
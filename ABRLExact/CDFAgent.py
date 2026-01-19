import time

from copy import deepcopy
from tqdm.auto import tqdm

import numpy as np
from scipy import stats

from ABRLExact.utils import sample_path_with_prob, add_obs, get_all_deterministic_paths, get_all_stochastic_paths



# def sample_path(obs, env, epsilon, sigma, all_paths, use_qmc=False, qmc_sobol_power=18, batch_size=1024, cache=None):
#     path_probs = []
#     internal_cache = cache if cache is not None else {}
#     for path in all_paths:
#         state_q_list, action_q_list = zip(*path)
#         prob = cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=state_q_list, action_q_list=action_q_list,
#                              normalise=False, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, batch_size=batch_size, cache=internal_cache)
#         path_probs.append(prob)

#     path_probs = np.array(path_probs)
#     path_probs_sum = np.sum(path_probs)
#     path_probs = path_probs / path_probs_sum

#     return sample_path_with_prob(all_paths, path_probs)

def sample_path(obs, env, epsilon, sigma, all_paths, use_qmc=False, qmc_sobol_power=18, batch_size=1024, cache=None):
    return _sample_path(obs=obs, env=env, epsilon=epsilon, sigma=sigma, all_paths=all_paths, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, batch_size=batch_size, cache=cache, bootstrap=None)


def sample_path_td(obs, env, epsilon, sigma, all_paths, use_qmc=False, qmc_sobol_power=18, bootstrap=None, num_bs_mc_samples=10000):
    return _sample_path(obs=obs, env=env, epsilon=epsilon, sigma=sigma, all_paths=all_paths, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, 
                        batch_size=None, cache=None, bootstrap=bootstrap, num_bs_mc_samples=num_bs_mc_samples)


def _sample_path(obs, env, epsilon, sigma, all_paths, use_qmc=False, qmc_sobol_power=18, batch_size=1024, cache=None, bootstrap=None, num_bs_mc_samples=10000):
    path_probs = []
    internal_cache = cache if cache is not None else {}
    for path in all_paths:
        state_q_list, action_q_list = zip(*path)
        if bootstrap is not None:
            prob, bootstrap = cdf_td_solution(obs=obs, env=env, epsilon=epsilon, sigma=sigma, bootstrap=bootstrap, state_q=None, state_q_list=state_q_list,  # pylint: disable=unbalanced-tuple-unpacking
                                              action_q_list=action_q_list, num_bs_mc_samples=num_bs_mc_samples, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power)
        else:
            prob = cdf_solution(obs=obs, env=env, epsilon=epsilon, sigma=sigma, state_q=None, state_q_list=state_q_list, action_q_list=action_q_list,
                                normalise=False, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, batch_size=batch_size, cache=internal_cache)
        path_probs.append(prob)

    path_probs = np.array(path_probs)
    path_probs_sum = np.sum(path_probs)
    path_probs = path_probs / path_probs_sum

    if bootstrap is not None:
        return sample_path_with_prob(all_paths, path_probs), bootstrap
    else:
        return sample_path_with_prob(all_paths, path_probs)


# https://github.com/scipy/scipy/blob/v1.16.1/scipy/stats/_qmc.py#L2312-L2483
def generate_base_sobol_normal(dim, power):
    engine = stats.qmc.Sobol(d=dim, scramble=True, bits=30)
    base_samples = engine.random(2**power)
    base_samples = stats.norm.ppf(0.5 + (1 - 1e-10) * (base_samples - 0.5))
    return base_samples

def mvn_cdf(lower, mean, cov, abseps=1e-5, use_qmc=True, qmc_sobol_power=18, qmc_base_cache=None):
    if not use_qmc:
        return stats.multivariate_normal.cdf(x=lower, mean=mean, cov=cov, abseps=abseps)
    else:
        if qmc_base_cache is None:
            sampler = stats.qmc.MultivariateNormalQMC(mean=mean, cov=cov)
            samples = sampler.random(2**qmc_sobol_power)
            return np.mean(np.all(samples < lower, axis=1))
        else:
            # https://github.com/scipy/scipy/blob/v1.16.1/scipy/stats/_qmc.py#L2312-L2483
            try:
                cov_root = np.linalg.cholesky(cov).transpose()
            except np.linalg.LinAlgError:
                eigval, eigvec = np.linalg.eigh(cov)
                if not np.all(eigval >= -1.0e-8):
                    raise ValueError("Covariance matrix not PSD.") from None
                eigval = np.clip(eigval, 0.0, None)
                cov_root = (eigvec * np.sqrt(eigval)).transpose()
            
            samples = qmc_base_cache[:,:len(mean)] @ cov_root + mean
            return np.mean(np.all(samples < lower, axis=1))
    
    
def qmc_configure_maxdim(ell_list, state_q, state_q_list, action_q_list, unique_state1_nongoal, permissible_actions_dict, mode):
    dim = 0
    for ell in ell_list:
        dim_tmp = 0
        for state_idx, state1 in enumerate(unique_state1_nongoal):
            for action in permissible_actions_dict[str(state1)]:
                action_ell = ell[state_idx]
                if not (action == action_ell):
                    dim_tmp += 1
        
        if mode == 0:
            for action_q in action_q_list:
                if state_q not in unique_state1_nongoal: # D_ell does not contain the new constraint yet
                    for action in permissible_actions_dict[str(state_q)]:
                        if not (action == action_q):
                            dim_tmp += 1
        elif mode == 1:
            for state_q_idx, state_q in enumerate(state_q_list): # compute probabilities for all state-action pairs in state_q_list and action_q_list
                action_q = action_q_list[state_q_idx]
                if state_q not in unique_state1_nongoal: # D_ell does not contain the new constraint yet
                    for action in permissible_actions_dict[str(state_q)]:
                        if not (action == action_q):
                            dim_tmp += 1

        dim = max(dim, dim_tmp)
    return dim


def qmc_configure_maxdim_td(state_q, state_q_list, action_q_list, permissible_actions_dict, mode):

    dim = 0
    if mode == 0:
        for action_q in action_q_list:
            for action in permissible_actions_dict[str(state_q)]:
                if not (action == action_q):
                    dim += 1
    elif mode == 1:
        for state_q_idx, state_q in enumerate(state_q_list): # compute probabilities for all state-action pairs in state_q_list and action_q_list
            action_q = action_q_list[state_q_idx]
            for action in permissible_actions_dict[str(state_q)]:
                if not (action == action_q):
                    dim += 1
                        
    return dim

    
def qmc_prep_helper(use_qmc, ell_list, state_q, state_q_list, action_q_list, unique_state1_nongoal, permissible_actions_dict, mode, power):
    if use_qmc:

        dim = qmc_configure_maxdim(ell_list=ell_list, 
                                   state_q=state_q, 
                                   state_q_list=state_q_list, 
                                   action_q_list=action_q_list, 
                                   unique_state1_nongoal=unique_state1_nongoal, 
                                   permissible_actions_dict=permissible_actions_dict, 
                                   mode=mode)
        qmc_base_cache = generate_base_sobol_normal(dim=dim, power=power)
    else:
        qmc_base_cache = None

    return qmc_base_cache


def qmc_prep_helper_td(use_qmc, state_q, state_q_list, action_q_list, permissible_actions_dict, mode, power):
    if use_qmc:

        dim = qmc_configure_maxdim_td(state_q=state_q, 
                                   state_q_list=state_q_list, 
                                   action_q_list=action_q_list, 
                                   permissible_actions_dict=permissible_actions_dict, 
                                   mode=mode)
        qmc_base_cache = generate_base_sobol_normal(dim=dim, power=power)
    else:
        qmc_base_cache = None

    return qmc_base_cache

    
    
def cdf_solution(obs, env, epsilon, sigma, state_q=None, state_q_list=None, action_q_list=None, abseps=1e-5, normalise=True, use_qmc=False, qmc_sobol_power=10, batch_size=1024, profile=False, cache=None):
    '''
    obs: dictionary of state0, action, state1, rewards
    env: initialised MDP class
    epsilon: likelihood variance
    sigma: prior variance
    state_q: If query is for the probability of optimality of actions of a given state_q, set state_q
    state_q_list: If query is for the probability of optimality of a sequence of state action pairs, set state_q_list and action_q_list instead, and keep state_q as None
    action_q_list: same as state_q_list
    cache: to be used for future runs when everything but state_q_list and action_q_list remain the same. If None, no cache will be used and no cache will be returned; else cache
           cached data will also be returned (incl. if cache is False). If cache is a dictionary containing "pr_vec", "pos_mean_vec", "pos_cov_vec", cache data will be used for computation
    '''
    
    

    if profile:
        tt = time.time() ###
    
    if state_q is not None and action_q_list is None:
        mode = 0
    else:
        assert state_q is None and state_q_list is not None and action_q_list is not None
        mode = 1
    
    n = len(obs["state0"])
    
    states = env.get_all_states()
    goal_states = env.get_all_terminal_states()

    # Get permissible action dictionary and parameter length
    theta_len = 0
    permissible_actions_dict = dict()
    for idx, state in enumerate(states):
        actions = env.get_possible_actions(state, gym_space=False)
        permissible_actions_dict[str(state)] = actions
        if state not in goal_states:
            theta_len += len(actions)

    if mode == 0:
        action_q_list = permissible_actions_dict[str(state_q)]

    # Get S^D, L^D,
    unique_state1 = set() # S
    for s in obs["state1"]:
        if isinstance(s, dict):
            unique_state1.update(s.keys())
        else:
            unique_state1.add(s)
    unique_state1 = list(unique_state1) # S^D + goal

    is_ell_list_empty = np.all([state in goal_states for state in unique_state1])

    if is_ell_list_empty:
        AprimeD_states = []
        AprimeD = []
        ell_list = np.array([[None]]) # L^D
    else:
        AprimeD_states, AprimeD = list(zip(*[[state, permissible_actions_dict[str(state)]] for state in unique_state1 if state not in goal_states]))
        ell_list = np.array(np.meshgrid(*AprimeD)).T.reshape(-1,len(AprimeD_states))
    
    unique_state1_nongoal = AprimeD_states # S^D

    unique_state1_nongoal_index = dict() # index dictionary of the list
    for idx, s in enumerate(unique_state1_nongoal):
        unique_state1_nongoal_index[s] = idx
        

    # preparation for second term
    flat_next_states = [] # flatted set of next_states involved in second term
    flat_row_idx = [] # flattedn set of corresponding row index of state0 involved in second term
    flat_probs = [] # flattened set of corresponding prob of next_state in second term
    flat_next_ell_idx = [] # flattened set of idx for ell lookup of next_actions in second term
    for i, state1 in enumerate(obs["state1"]):
        transitions = state1 if isinstance(state1, dict) else {state1: 1.}

        for s1, p in transitions.items():
            if s1 not in goal_states:
                flat_next_states.append(s1)
                flat_row_idx.append(i)
                flat_probs.append(p)
                flat_next_ell_idx.append(unique_state1_nongoal_index[s1])
    flat_next_states = np.array(flat_next_states)
    flat_row_idx = np.array(flat_row_idx)
    flat_probs = np.array(flat_probs)
    flat_next_ell_idx = np.array(flat_next_ell_idx)

    num_batches = int(np.ceil(len(ell_list) / batch_size))
    
    # cache to be reused if all but state_q_list and action_q_list remain the same
    cached_data = False
    full_pos_mean_list = None
    full_pos_cov_list = None
    full_pos_mean = None
    full_pos_cov = None
    if cache is not None and "pos_mean_array" in cache and "pos_cov_array" in cache and "pr_vec" in cache:
        cached_data = True
        pr_vec = cache["pr_vec"]
        full_pos_mean = cache["pos_mean_array"]
        full_pos_cov = cache["pos_cov_array"]
    else:
        pr_vec = np.zeros(len(ell_list))
        if cache is not None:
            full_pos_mean_list = []
            full_pos_cov_list = []

    # Get pEr and pEsr
    pEr_vec = np.zeros(len(ell_list))
    if mode == 0:
        pEstarr_list = [np.zeros(len(ell_list)) for i in range(len(action_q_list))]
    elif mode == 1:
        pEstarr_vec = np.zeros(len(ell_list))
        
    # qmc prep
    qmc_base_cache = qmc_prep_helper(use_qmc=use_qmc, 
                                      ell_list=ell_list, 
                                      state_q=state_q, 
                                      state_q_list=state_q_list, 
                                      action_q_list=action_q_list, 
                                      unique_state1_nongoal=unique_state1_nongoal, 
                                      permissible_actions_dict=permissible_actions_dict, 
                                      mode=mode,
                                     power=qmc_sobol_power)

    
    if profile:
        print(f"Zone A {time.time() - tt}") ###
        tt = time.time() ###

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(ell_list))

        if not cached_data:
            B_ell_list = []
            for idx in range(start_idx, end_idx):
                ell = ell_list[idx]
                
                # first term
                B_ell = np.zeros([n, theta_len])
                if n > 0:
                    B_ell[np.arange(n), np.array(env.nu_vectorised(state=obs["state0"], action=obs["action"])) - 1] += 1

                # second term
                if len(flat_row_idx) > 0:
                    flat_col_idx = np.array(env.nu_vectorised(state=flat_next_states, action=ell[flat_next_ell_idx])) - 1
                    np.add.at(B_ell, (flat_row_idx, flat_col_idx), - flat_probs)

                B_ell_list.append(B_ell)

            if profile:
                print(f"Zone B {time.time() - tt} at batch {i}/{num_batches}") ###
                tt = time.time() ###

                    
            if len(obs["rewards"]) > 0:
                B_batch = np.stack(B_ell_list)
                
                Gamma_inv_batch = (sigma**2) * (B_batch @ B_batch.transpose(0, 2, 1)) + (epsilon**2) * np.eye(n)
                Gamma_batch = np.linalg.inv(Gamma_inv_batch)

                _, log_det = np.linalg.slogdet(Gamma_inv_batch)
                quad_term = (Gamma_batch @ obs["rewards"] * obs["rewards"]).sum(axis=1)
                pr_vec[start_idx:end_idx] = np.exp( -0.5 * (n * np.log(2 * np.pi) + log_det + quad_term)) # p^ell(r1:n)

                #Get p(theta|r)
                pos_cov_ = B_batch.transpose(0, 2, 1) @ Gamma_batch
                pos_mean_array = (sigma**2) * pos_cov_  @ obs["rewards"] # posterior mean of p^ell(theta|r1:n)
                pos_cov_array = (sigma**2) * np.eye(theta_len) - (sigma**4) * (pos_cov_ @ B_batch) # posterior covariance of p^ell(theta|r1:n)
                
            else:
                pos_mean_array = np.zeros((1, theta_len))
                pos_cov_array = np.array([sigma**2 * np.eye(theta_len)])
                pr_vec[0] = 1.
            
            if cache is not None:
                full_pos_mean_list.append(pos_mean_array)
                full_pos_cov_list.append(pos_cov_array)
        else:
            pos_mean_array = full_pos_mean[start_idx:end_idx]
            pos_cov_array = full_pos_cov[start_idx:end_idx]

        if profile:
            print(f"Zone C {time.time() - tt} at batch {i}/{num_batches}") ###
            tt = time.time() ###

        
        # Get Eell transformation matrix
        D_ell_list = []
        if mode == 0:
            D_ell_star_list = [[] for i in range(len(action_q_list))]
        elif mode == 1:
            D_ell_star_list = []

        for idx in range(start_idx, end_idx):
            ell = ell_list[idx]

            D_ell = [] # matrix transformation for cdf computation of E^ell
            D_ell_star = [] # matrix transformation for cdf computation of E^*
            for state_idx, state1 in enumerate(unique_state1_nongoal):
                for action in permissible_actions_dict[str(state1)]:
                    action_ell = ell[state_idx]
                    if not (action == action_ell):
                        D_ell_tmp =  np.zeros(theta_len)
                        D_ell_tmp[env.nu(state=state1, action=action)-1] = 1
                        D_ell_tmp[env.nu(state=state1, action=action_ell)-1] = -1
                        D_ell.append(D_ell_tmp)

            if mode == 0:
                for action_idx, action_q in enumerate(action_q_list): # compute probabilities for all actions
                    D_ell_star = deepcopy(D_ell)
                    if state_q not in unique_state1_nongoal: # D_ell does not contain the new constraint yet
                        for action in permissible_actions_dict[str(state_q)]:
                            if not (action == action_q):
                                D_ell_star_tmp = np.zeros(theta_len)
                                D_ell_star_tmp[env.nu(state=state_q, action=action)-1] = 1
                                D_ell_star_tmp[env.nu(state=state_q, action=action_q)-1] = -1
                                D_ell_star.append(D_ell_star_tmp)
        
                    else: # D_ell contains the new constraint, check for contradictions
                        state_q_idx = unique_state1_nongoal_index[state_q]
                        if ell[state_q_idx] == action_q: # E* and Ell align, set D_ell_star as D_ell
                            pass
                        else: # otherwise, set it as None, to be dealt with later
                            D_ell_star = None
                    
                    D_ell_star = np.array(D_ell_star)
                    D_ell_star_list[action_idx].append(D_ell_star)

            elif mode == 1:
                D_ell_star = deepcopy(D_ell)
                for state_q_idx, state_q in enumerate(state_q_list): # compute probabilities for all state-action pairs in state_q_list and action_q_list
                    action_q = action_q_list[state_q_idx]
                    if state_q not in unique_state1_nongoal: # D_ell does not contain the new constraint yet
                        for action in permissible_actions_dict[str(state_q)]:
                            if not (action == action_q):
                                D_ell_star_tmp = np.zeros(theta_len)
                                D_ell_star_tmp[env.nu(state=state_q, action=action)-1] = 1
                                D_ell_star_tmp[env.nu(state=state_q, action=action_q)-1] = -1
                                D_ell_star.append(D_ell_star_tmp)
        
                    else: # D_ell contains the new constraint, check for contradictions
                        state_q_idx = unique_state1_nongoal_index[state_q]
                        if ell[state_q_idx] == action_q:
                            pass
                        else:
                            D_ell_star = None
                            break
        
                D_ell_star = np.array(D_ell_star)
                D_ell_star_list.append(D_ell_star)


            D_ell = np.array(D_ell)
            D_ell_list.append(D_ell)
        
        
        if profile:
            print(f"Zone D {time.time() - tt} at batch {i}/{num_batches}") ###
            tt = time.time() ###

        for idx in range(start_idx, end_idx):
            ell = ell_list[idx]
            D_ell = D_ell_list[idx-start_idx]

            if normalise:
                if len(D_ell) == 0:
                    pEr_vec[idx] = 1.
                else:
                    pEr_mean = D_ell @ pos_mean_array[idx-start_idx] # transformed posterior mean for cdf for E^ell
                    pEr_cov = D_ell @ pos_cov_array[idx-start_idx] @ D_ell.T # transformed posterior covariance for cdf for E^ell
                    pEr_vec[idx] = mvn_cdf(lower=np.zeros(len(D_ell)), mean=pEr_mean, cov=pEr_cov, abseps=abseps, 
                                        use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, qmc_base_cache=qmc_base_cache)

            if mode == 0:
                for action_idx in range(len(action_q_list)):
                    D_ell_star = D_ell_star_list[action_idx][idx-start_idx]

                    if (not D_ell_star.shape == ()) and (len(D_ell_star) > 0): # if it is not np.array(None)
                        pEstarr_mean = D_ell_star @ pos_mean_array[idx-start_idx] # transformed posterior mean for cdf for E^*
                        pEstarr_cov = D_ell_star @ pos_cov_array[idx-start_idx] @ D_ell_star.T # transformed posterior covariance for cdf for E^*
                        pEstarr_list[action_idx][idx] = mvn_cdf(lower=np.zeros(len(D_ell_star)), mean=pEstarr_mean, cov=pEstarr_cov, abseps=abseps, 
                                                                use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, qmc_base_cache=qmc_base_cache)
                    elif D_ell_star.shape == (0,): # no constraint, it is np.array([]), so return the probability of the sample space, which is 1
                        pEstarr_list[action_idx][idx] = 1.
                    else: #if it is np.array(None), prob is 0
                        pEstarr_list[action_idx][idx] = 0.
            elif mode == 1:
                D_ell_star = D_ell_star_list[idx-start_idx]
                if (not D_ell_star.shape == ()) and (len(D_ell_star) > 0): #if it is not np.array(None)
                    pEstarr_mean = D_ell_star @ pos_mean_array[idx-start_idx]
                    pEstarr_cov = D_ell_star @ pos_cov_array[idx-start_idx] @ D_ell_star.T
                    pEstarr_vec[idx] = mvn_cdf(lower=np.zeros(len(D_ell_star)), mean=pEstarr_mean, cov=pEstarr_cov, abseps=abseps, 
                                            use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, qmc_base_cache=qmc_base_cache)
                elif D_ell_star.shape == (0,): # no constraint
                    pEstarr_vec[idx] = 1.
                else: #if it is np.array(None), prob is 0
                    pEstarr_vec[idx] = 0.


    if profile:
        print(f"Zone E {time.time() - tt}") ###
        
    if cache is not None and not cached_data:
        cache["pos_mean_array"] = np.concatenate(full_pos_mean_list, axis=0)
        cache["pos_cov_array"] = np.concatenate(full_pos_cov_list, axis=0)
        cache["pr_vec"] = pr_vec

    # Get final answer
    if mode == 0:
        pEstarr_numer = np.array([np.sum(pr_vec * pEstarr_vec) for pEstarr_vec in pEstarr_list])
        if normalise:
            pEstarr_denom = np.sum(pr_vec * pEr_vec)
            pEstarr = pEstarr_numer / pEstarr_denom
        else:
            pEstarr = pEstarr_numer / np.sum(pEstarr_numer)
        return pEstarr, action_q_list

    elif mode == 1:
        pEstarr_numer = np.sum(pr_vec * pEstarr_vec)
        if normalise:
            pEstarr_denom = np.sum(pr_vec * pEr_vec)
            pEstarr = pEstarr_numer / pEstarr_denom
        else:
            pEstarr = pEstarr_numer
        return pEstarr



def expected_q_max_mc(mean, cov, env, obs, n_samples=100000):

    samples = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples)
    goal_states = env.get_all_terminal_states()

    next_states_set = set()
    for i, state1 in enumerate(obs["state1"]):
        if isinstance(state1, dict):
            for s1, p in state1.items():
                if s1 not in goal_states:
                    next_states_set.add(s1)
        else:
            if state1 not in goal_states:
                next_states_set.add(state1)

    max_dict = {}
    for s1 in next_states_set:
        actions = np.array(env.get_possible_actions(state=s1, gym_space=False))
        indices = np.array(env.nu_vectorised(state=[s1], action=actions)) - 1
        max_dict[s1] = np.mean(np.max(samples[:, indices], axis=1))
        
    expected_max = np.zeros(len(obs["state0"]))

    for i, state1 in enumerate(obs["state1"]):
        transitions = state1 if isinstance(state1, dict) else {state1: 1.}
        for s1, p in transitions.items():
            if s1 not in goal_states:
                expected_max[i] += p * max_dict[s1]

    return expected_max


def cdf_td_solution(obs, env, epsilon, sigma, bootstrap, state_q=None, state_q_list=None, action_q_list=None, abseps=1e-5, num_bs_mc_samples=10000, use_qmc=False, qmc_sobol_power=10, profile=False):
    if profile:
        tt = time.time() ###
    
    if state_q is not None and action_q_list is None:
        mode = 0
    else:
        assert state_q is None and state_q_list is not None and action_q_list is not None
        mode = 1

    n = len(obs["state0"])
    
    states = env.get_all_states()
    goal_states = env.get_all_terminal_states()

    # Get permissible action dictionary and parameter length
    theta_len = 0
    permissible_actions_dict = dict()
    for state in states:
        actions = env.get_possible_actions(state, gym_space=False)
        permissible_actions_dict[str(state)] = actions
        if state not in goal_states:
            theta_len += len(actions)

    if mode == 0:
        action_q_list = permissible_actions_dict[str(state_q)]

    qmc_base_cache = qmc_prep_helper_td(use_qmc=use_qmc, 
                                        state_q=state_q, 
                                        state_q_list=state_q_list, 
                                        action_q_list=action_q_list, 
                                        permissible_actions_dict=permissible_actions_dict, 
                                        mode=mode,
                                        power=qmc_sobol_power)
    
    if profile:
        print(f"Zone A {time.time() - tt}") ###
        tt = time.time() ###

    B = np.zeros([n, theta_len])
    if n > 0:
        B[np.arange(n), np.array(env.nu_vectorised(state=obs["state0"], action=obs["action"])) - 1] += 1

    if profile:
        print(f"Zone B {time.time() - tt}") ###
        tt = time.time() ###

    if len(obs["rewards"]) > 0:

        b_mean, b_cov = bootstrap
        expected_q = expected_q_max_mc(mean=b_mean, cov=b_cov, env=env, obs=obs, n_samples=num_bs_mc_samples)
        
        Gamma_inv = (sigma**2) * (B @ B.transpose()) + (epsilon**2) * np.eye(n)
        Gamma = np.linalg.inv(Gamma_inv)

        #Get p(theta|r)
        pos_cov_ = B.transpose() @ Gamma
        pos_mean = (sigma**2) * pos_cov_  @ (obs["rewards"] + expected_q) # posterior mean of p(theta|r1:n)
        pos_cov = (sigma**2) * np.eye(theta_len) - (sigma**4) * (pos_cov_ @ B) # posterior covariance of p(theta|r1:n)
        
    else:
        pos_mean = np.zeros(theta_len)
        pos_cov = sigma**2 * np.eye(theta_len)

    if profile:
        print(f"Zone C {time.time() - tt}") ###
        tt = time.time() ###


    if mode == 0:
        D_star_list = []
        for action_idx, action_q in enumerate(action_q_list): # compute probabilities for all actions
            D_star = []
            for action in permissible_actions_dict[str(state_q)]:
                if not (action == action_q):
                    D_star_tmp = np.zeros(theta_len)
                    D_star_tmp[env.nu(state=state_q, action=action)-1] = 1
                    D_star_tmp[env.nu(state=state_q, action=action_q)-1] = -1
                    D_star.append(D_star_tmp)
            
            D_star = np.array(D_star)
            D_star_list.append(D_star)

        if profile:
            print(f"Zone D {time.time() - tt}") ###
            tt = time.time() ###

        pEstarr_array = np.zeros(len(action_q_list))
        for action_idx in range(len(action_q_list)):
            D_star = D_star_list[action_idx]

            if (not D_star.shape == ()) and (len(D_star) > 0): # if it is not np.array(None)
                pEstarr_mean = D_star @ pos_mean # transformed posterior mean for cdf for E^*
                pEstarr_cov = D_star @ pos_cov @ D_star.T # transformed posterior covariance for cdf for E^*
                pEstarr_array[action_idx] = mvn_cdf(lower=np.zeros(len(D_star)), mean=pEstarr_mean, cov=pEstarr_cov, abseps=abseps, 
                                                        use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, qmc_base_cache=qmc_base_cache)
            elif D_star.shape == (0,): # no constraint, it is np.array([]), so return the probability of the sample space, which is 1
                pEstarr_array[action_idx] = 1.
            else: #if it is np.array(None), prob is 0
                pEstarr_array[action_idx] = 0.

    elif mode == 1:
        D_star = []
        for state_q_idx, state_q in enumerate(state_q_list): # compute probabilities for all state-action pairs in state_q_list and action_q_list
            action_q = action_q_list[state_q_idx]
            for action in permissible_actions_dict[str(state_q)]:
                if not (action == action_q):
                    D_star_tmp = np.zeros(theta_len)
                    D_star_tmp[env.nu(state=state_q, action=action)-1] = 1
                    D_star_tmp[env.nu(state=state_q, action=action_q)-1] = -1
                    D_star.append(D_star_tmp)

        D_star = np.array(D_star)

        if profile:
            print(f"Zone D {time.time() - tt}") ###
            tt = time.time() ###

        if (not D_star.shape == ()) and (len(D_star) > 0): #if it is not np.array(None)
            pEstarr_mean = D_star @ pos_mean
            pEstarr_cov = D_star @ pos_cov @ D_star.T
            pEstarr = mvn_cdf(lower=np.zeros(len(D_star)), mean=pEstarr_mean, cov=pEstarr_cov, abseps=abseps, 
                                    use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, qmc_base_cache=qmc_base_cache)
        elif D_star.shape == (0,): # no constraint
            pEstarr = 1.
        else: #if it is np.array(None), prob is 0
            pEstarr = 0.

    if profile:
        print(f"Zone E {time.time() - tt}") ###

    if mode == 0:
        return pEstarr_array, action_q_list, (pos_mean, pos_cov)
    else:
        return pEstarr, (pos_mean, pos_cov)
    
    
def run_cdf_deepsea(env, epsilon=0.02, sigma=10, num_episodes=30, use_qmc=False, qmc_sobol_power=18, output_obs=False, batch_size=1024, save_path=None, disable_tqdm=False, tqdm_position=None):
    return _run_cdf_deepsea(env=env, epsilon=epsilon, sigma=sigma, num_episodes=num_episodes, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, output_obs=output_obs, batch_size=batch_size, save_path=save_path, use_bootstrap=False, disable_tqdm=disable_tqdm, tqdm_position=tqdm_position)


def run_cdf_deepsea_td(env, epsilon=0.02, sigma=10, num_episodes=30, use_qmc=False, qmc_sobol_power=18, output_obs=False, num_bs_mc_samples=10000, save_path=None, disable_tqdm=False, tqdm_position=None):
    return _run_cdf_deepsea(env=env, epsilon=epsilon, sigma=sigma, num_episodes=num_episodes, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, output_obs=output_obs, use_bootstrap=True, num_bs_mc_samples=num_bs_mc_samples, save_path=save_path, disable_tqdm=disable_tqdm, tqdm_position=tqdm_position)


def _run_cdf_deepsea(env, epsilon=0.02, sigma=10, num_episodes=30, use_qmc=False, qmc_sobol_power=18, output_obs=False, batch_size=1024, use_bootstrap=False, num_bs_mc_samples=10000, save_path=None, disable_tqdm=False, tqdm_position=None):

    # get all paths
    if env.deterministic_transition is True:
        all_paths, _ = get_all_deterministic_paths(env)
    else:
        all_paths = get_all_stochastic_paths(env)

    # store history
    unique_stat = set()
    obs = {"state0": [], "action": [], "state1": [], "rewards": [], "done": []}
    state_history = [[]]
    reward_history = [[]]
    all_probs = [] # store marginal prob of state optimality
    all_optimal_path_probs = [] # store optimal path probabilities
    if output_obs is True:
        obs_history = [deepcopy(obs)]

    # initialise
    if use_bootstrap:
        theta_len = 0
        states = env.get_all_states()
        goal_states = env.get_all_terminal_states()
        for state in states:
            actions = env.get_possible_actions(state, gym_space=False)
            if state not in goal_states:
                theta_len += len(actions)
        bootstrap = [np.zeros(theta_len), np.eye(theta_len) * sigma ** 2]

    probs_vec = [] 
    obs_cache = {}
    for s in env.get_all_states(): # compute initial marginal prob of state optimality
        if s not in env.get_all_terminal_states():
            if use_bootstrap:
                res = cdf_td_solution(obs=obs, env=env, epsilon=epsilon, sigma=sigma, bootstrap=bootstrap, state_q=s, use_qmc=use_qmc, num_bs_mc_samples=num_bs_mc_samples, qmc_sobol_power=qmc_sobol_power)
            else:
                res = cdf_solution(obs=obs, env=env, epsilon=epsilon, sigma=sigma, state_q=s, normalise=False, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, batch_size=batch_size, cache=obs_cache)
            probs_vec.append(res[0][0])
    all_probs.append(probs_vec)

    new_flag = True # store flag of whether new observations have been added
    

    # main loop
    tqdm_desc = f"Worker {tqdm_position}" if tqdm_position is not None else None
    for _ in tqdm(range(num_episodes), disable=disable_tqdm, position=tqdm_position, desc=tqdm_desc, leave=False):
        curr_state = (0, 0)
        history = [curr_state]
        reward_acc = []

        # rollout
        if new_flag is True or use_bootstrap:
            if use_bootstrap:
                # return bootstrap as a conclusion of the posterior given the current observations, to be used for the next iteration after observations are updated
                (policy, path_probs), bootstrap = sample_path_td(obs=obs, env=env, epsilon=epsilon, sigma=sigma, bootstrap=bootstrap, all_paths=all_paths, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, num_bs_mc_samples=num_bs_mc_samples)
            else:
                policy, path_probs = sample_path(obs=obs, env=env, epsilon=epsilon, sigma=sigma, all_paths=all_paths, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, batch_size=batch_size, cache=obs_cache)
        else:
            policy, path_probs = sample_path_with_prob(all_paths, path_probs)
        all_optimal_path_probs.append(path_probs)

        new_flag = False
        done = False
        while done is False:

            action = policy[curr_state] 
            next_state, reward, done = env.step(action=action, is_simulation=True, simulation_state=curr_state)
            next_state_obs = next_state if env.deterministic_transition else env.transition_distribution(state=curr_state, action=action)
            _, _, flag = add_obs(obs=obs, unique_stat=unique_stat, state0=curr_state, action=action, state1=next_state_obs, reward=reward, done=done)
            curr_state = next_state
            reward_acc.append(reward)
            new_flag = new_flag or flag

            if env.deterministic_transition is True and next_state in history: #loop exit
                history.append(next_state)
                break
            history.append(next_state)
            
        state_history.append(history)
        reward_history.append(reward_acc)
        if output_obs is True:
            obs_history.append(deepcopy(obs))

        # compute marginal prob of state optimality
        if new_flag is True or use_bootstrap:
            obs_cache = {}
            mprobs_vec = []
            for s in env.get_all_states():
                if s not in env.get_all_terminal_states():
                    if use_bootstrap:
                        res = cdf_td_solution(obs=obs, env=env, epsilon=epsilon, sigma=sigma, bootstrap=bootstrap, state_q=s, use_qmc=use_qmc, num_bs_mc_samples=num_bs_mc_samples, qmc_sobol_power=qmc_sobol_power)
                    else:
                        res = cdf_solution(obs, env, epsilon, sigma, state_q=s, normalise=False, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, batch_size=batch_size, cache=obs_cache)
                    mprobs_vec.append(res[0][0])
        all_probs.append(mprobs_vec)

    # final optimal path prob
    if new_flag is True or use_bootstrap:
        if use_bootstrap:
                (policy, path_probs), bootstrap = sample_path_td(obs=obs, env=env, epsilon=epsilon, sigma=sigma, bootstrap=bootstrap, 
                                                                 all_paths=all_paths, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, num_bs_mc_samples=num_bs_mc_samples)
        else:
            policy, path_probs = sample_path(obs=obs, env=env, epsilon=epsilon, sigma=sigma, all_paths=all_paths, use_qmc=use_qmc, qmc_sobol_power=qmc_sobol_power, batch_size=batch_size, cache=obs_cache)
    else:
        policy, path_probs = sample_path_with_prob(all_paths, path_probs)
    all_optimal_path_probs.append(path_probs)

    if save_path is not None:
        save_dict = {
            "state_history": state_history,
            "probs_data": (all_probs, env.action_map),
            "paths_data": (all_paths, all_optimal_path_probs, env.deterministic_transition),
            "reward_history": reward_history
        }
        if output_obs:
            save_dict["obs_history"] = obs_history
        
        np.save(save_path, save_dict)

    if output_obs:
        return state_history, (all_probs, env.action_map), (all_paths, all_optimal_path_probs, env.deterministic_transition), reward_history, obs_history
    else:
        return state_history, (all_probs, env.action_map), (all_paths, all_optimal_path_probs, env.deterministic_transition), reward_history
from copy import deepcopy
from tqdm.auto import tqdm

import numpy as np
import torch

import pyro
import pyro.distributions as ndist
from pyro.infer import HMC

from ABRLExact.utils import get_default_device, add_obs


class pyro_model_deepsea:
    def __init__(self, parameter_dim, env, prior_std=1., epsilon=0.1, device=None):
        self.parameter_dim = parameter_dim
        self.env = env
        self.prior_std = prior_std
        self.epsilon = epsilon

        self.device = torch.device(device) if device is not None else get_default_device()


    def _get_expected_q(self, parameter, done_batch, next_states_batch, trans_probs_batch):
        possible_actions = torch.arange(self.env.action_space.n, device=self.device)

        next_states_batch = next_states_batch.unsqueeze(2) # (batch, num_possible_states, 1, 2)
        possible_actions_batch = possible_actions.view(1, 1, -1) # (1, 1, num_actions)

        done_mask = done_batch.view(-1, 1, 1).bool()
        tmp_indices = self.env.nu_torch(next_states_batch, possible_actions_batch) - 1 # avoid out-of-bound
        indices = torch.where(done_mask, torch.zeros_like(tmp_indices), tmp_indices)
        tmp_next_q = parameter[indices]
        next_q = torch.where(done_mask, torch.zeros_like(tmp_next_q), tmp_next_q)
        
        next_v = next_q.max(dim=-1).values # (batch, num_possible_states)

        expected_q = (next_v * trans_probs_batch).sum(dim=1) # (batch)
        
        return expected_q

    def preprocess_obs(self, obs):
        state0 = torch.tensor(np.array(obs["state0"]), dtype=torch.long, device=self.device)
        action = torch.tensor(obs["action"], dtype=torch.long, device=self.device)
        reward = torch.tensor(obs["rewards"], dtype=torch.float32, device=self.device)
        done = torch.tensor(obs["done"], dtype=torch.long, device=self.device)

        if isinstance(obs["state1"][0], dict):
            next_state_list = [list(state.keys()) for state in obs["state1"]]
            trans_probs_list = [list(state.values()) for state in obs["state1"]]

            next_states = torch.tensor(np.array(next_state_list), dtype=torch.long, device=self.device)
            trans_probs = torch.tensor(np.array(trans_probs_list), dtype=torch.float32, device=self.device)
        else:
            next_states = torch.tensor(np.array(obs["state1"]), dtype=torch.long, device=self.device).unsqueeze(1)
            trans_probs = torch.tensor([1.], dtype=torch.float32, device=self.device)

        data = {"state0": state0, "action": action, "reward": reward, "done": done, "next_states": next_states, "trans_probs": trans_probs}
        return data
        
        
    def pyro_model(self, data):
        parameter = pyro.sample("prior_parameter", ndist.Normal(torch.zeros(self.parameter_dim, device=self.device), self.prior_std).to_event(1))

        state_batch = data["state0"].to(self.device)
        action_batch = data["action"].to(self.device)
        done_batch = data["done"].to(self.device)
        next_states_batch = data["next_states"].to(self.device)
        trans_probs_batch = data["trans_probs"].to(self.device)
        reward_batch = data["reward"].to(self.device)

        n = len(state_batch)

        curr_q = parameter[self.env.nu_torch(state_batch, action_batch) - 1]
        expected_next_q = self._get_expected_q(parameter=parameter, 
                                               done_batch=done_batch,
                                               next_states_batch=next_states_batch,
                                               trans_probs_batch=trans_probs_batch)

        mean = curr_q - torch.where(done_batch == 1, torch.zeros_like(expected_next_q), expected_next_q)

        with pyro.plate("data", n):
            pyro.sample("obs", ndist.Normal(loc=mean, scale=self.epsilon), obs=reward_batch)


    def pyro_model_tr(self, data, diag_inv_mass_sqrt):
        parameter_tr = pyro.sample("prior_parameter_tr", ndist.Normal(torch.zeros(self.parameter_dim, device=self.device), self.prior_std / diag_inv_mass_sqrt).to_event(1))
                                                                                                                                  
        parameter = parameter_tr * diag_inv_mass_sqrt

        state_batch = data["state0"].to(self.device)
        action_batch = data["action"].to(self.device)
        done_batch = data["done"].to(self.device)
        next_states_batch = data["next_states"].to(self.device)
        trans_probs_batch = data["trans_probs"].to(self.device)
        reward_batch = data["reward"].to(self.device)

        n = len(state_batch)

        curr_q = parameter[self.env.nu_torch(state_batch, action_batch) - 1]
        expected_next_q = self._get_expected_q(parameter=parameter,
                                               done_batch=done_batch,
                                               next_states_batch=next_states_batch,
                                               trans_probs_batch=trans_probs_batch)

        mean = curr_q - expected_next_q


        with pyro.plate("data", n):
            pyro.sample("obs", ndist.Normal(loc=mean, scale=self.epsilon), obs=reward_batch)
            
            
def warmup(model, data, initial_step_size, num_steps, num_samples_per_run, num_runs, initial_params, param_name="prior_parameter", diag_inv_mass_sqrt=1., target_acc_prob=0.75, 
           step_size_rates=(1.3, 0.7, 1.1, 0.8), disable_progbar=False):
    
    step_size = initial_step_size
    current_params = initial_params

    last_safe_step_size = None
    last_safe_params = None

    finetune_phase = False
    last_action = None
    
    for _ in range(num_runs):
        kernel = HMC(model=model, step_size=step_size, num_steps=num_steps, adapt_step_size=False, 
                     adapt_mass_matrix=False, trajectory_length=None)

        pyro_mcmc = pyro.infer.mcmc.MCMC(kernel=kernel, 
                                    num_samples=num_samples_per_run,
                                    num_chains=1,
                                    initial_params={param_name:initial_params},
                                    warmup_steps=0,
                                    disable_progbar=disable_progbar)

        if diag_inv_mass_sqrt is None:
            pyro_mcmc.run(data=data)
        else:
            pyro_mcmc.run(data=data, diag_inv_mass_sqrt=diag_inv_mass_sqrt)
        final_params = pyro_mcmc.get_samples()[param_name][-1].detach()
        
        accept_prob = pyro_mcmc.diagnostics()["acceptance rate"]["chain 0"]
        if accept_prob < target_acc_prob: # decrease step size
            current_action = "decrease"
            if accept_prob < 0.1 and last_safe_params is not None:
                current_params = last_safe_params
            else:
                current_params = final_params

            factor = step_size_rates[3] if finetune_phase else step_size_rates[1]
        else: # increase step size
            current_action = "increase"

            last_safe_step_size = step_size
            last_safe_params = final_params
            current_params = final_params

            factor = step_size_rates[2] if finetune_phase else step_size_rates[0]

        if last_action is not None and current_action != last_action:
            if not finetune_phase:
                finetune_phase = True

        step_size *= factor
        last_action = current_action

    if last_safe_step_size is not None:
        return last_safe_step_size, last_safe_params
    else:
        return step_size, current_params



def sample_q_mcmc(obs, env, epsilon, sigma, num_samples, step_size, num_steps, num_warmup_runs=0, num_warmup_samples_per_run=10, 
                  fitted_diag_std=None, target_acc_prob=0.75, step_size_rates=(1.3, 0.7, 1.1, 0.8), disable_progbar=False):

    parameter_dim = (len(env.get_all_states()) - len(env.get_all_terminal_states())) * 2

    pyro_model_cls = pyro_model_deepsea(parameter_dim=parameter_dim, env=env, prior_std=sigma, epsilon=epsilon)
    proprocessed_data = pyro_model_cls.preprocess_obs(obs)

    if fitted_diag_std is None:
        pyro_model = pyro_model_cls.pyro_model
        initial_params = torch.randn(parameter_dim) * sigma
        param_name = "prior_parameter"
    else:
        pyro_model = pyro_model_cls.pyro_model_tr
        initial_params = torch.randn(parameter_dim) * sigma / fitted_diag_std
        param_name = "prior_parameter_tr"

    if num_warmup_runs > 0:
        step_size, initial_params = warmup(model=pyro_model, data=proprocessed_data, initial_step_size=step_size, num_steps=num_steps, 
                                           num_samples_per_run=num_warmup_samples_per_run, num_runs=num_warmup_runs, initial_params=initial_params, 
                                           param_name=param_name, diag_inv_mass_sqrt=fitted_diag_std, target_acc_prob=target_acc_prob, 
                                           step_size_rates=step_size_rates, disable_progbar=disable_progbar)

    kernel = HMC(model=pyro_model, step_size=step_size, num_steps=num_steps, adapt_step_size=False, 
                 adapt_mass_matrix=False, trajectory_length=None)

    pyro_mcmc = pyro.infer.mcmc.MCMC(kernel=kernel, 
                                     num_samples=num_samples,
                                     num_chains=1,
                                     initial_params={param_name:initial_params},
                                     warmup_steps=0,
                                    disable_progbar=disable_progbar)
        
    if fitted_diag_std is None:
        pyro_mcmc.run(data=proprocessed_data)
        all_samples = pyro_mcmc.get_samples()[param_name]
    else:
        pyro_mcmc.run(data=proprocessed_data, diag_inv_mass_sqrt=fitted_diag_std)
        all_samples = pyro_mcmc.get_samples()[param_name] * fitted_diag_std

    samples = all_samples[-1]

    step_size = pyro_mcmc.kernel.step_size
    accept_prob = pyro_mcmc.diagnostics()["acceptance rate"]["chain 0"]
    return samples.numpy(), step_size, all_samples.numpy(), accept_prob


def run_hmc_deepsea(env, epsilon=0.02, sigma=10, num_episodes=30, num_samples=100, 
                    step_size=0.01, num_steps=10, num_warmup_runs=10, num_warmup_samples_per_run=10, 
                    target_acc_prob=0.75, step_size_rates=(1.3, 0.7, 1.1, 0.8), disable_progbar=False, 
                    input_obs=None, disable_tqdm=False, tqdm_position=None, output_obs=False, save_path=None):

    unique_stat = set()
    obs = {"state0": [], "action": [], "state1": [], "rewards": [], "done": []}

    state_history = [[]]
    reward_history = [[]]
    parameter_dim = (len(env.get_all_states()) - len(env.get_all_terminal_states())) * 2
    samples_history = []
    acc_prob_history = []
    if output_obs is True:
        obs_history = [deepcopy(obs)]
    
    if input_obs is not None: # for experimental use where obs is guided by an external algorithm (e.g. cdf)
        num_episodes = len(input_obs)

    
    tqdm_desc = f"Worker {tqdm_position}" if tqdm_position is not None else None
    for i in tqdm(range(num_episodes), disable=disable_tqdm, position=tqdm_position, desc=tqdm_desc, leave=False):
        curr_state = (0, 0)
        history = [curr_state]
        reward_acc = []

        if input_obs is not None: 
            obs = input_obs[i]
            
        if output_obs is True: 
            obs_history.append(deepcopy(obs))
    
        if i > 0 or input_obs is not None:
            if i == 0:
                fitted_diag_std = 1.
            else:
                k = min(1, len(all_q_samples) // 2)
                fitted_diag_std = torch.tensor(np.std(all_q_samples[k:], axis=0, ddof=1) + 1e-3) # to ensure it is non-zero
            q_sample, step_size, all_q_samples, acc_prob = sample_q_mcmc(obs=obs, env=env, epsilon=epsilon, sigma=sigma, num_samples=num_samples, 
                                                               step_size=step_size, num_steps=num_steps, num_warmup_runs=num_warmup_runs,
                                                               num_warmup_samples_per_run=num_warmup_samples_per_run, fitted_diag_std=fitted_diag_std, 
                                                               target_acc_prob=target_acc_prob, step_size_rates=step_size_rates, 
                                                               disable_progbar=disable_progbar)

        else:
            all_q_samples = np.random.randn(num_samples, parameter_dim) * sigma
            q_sample = all_q_samples[0]
            acc_prob = 1.
            
        acc_prob_history.append(acc_prob)
        
        done = False
        while done is False:
            action =  np.argmax(q_sample[env.nu_vectorised(state=[curr_state], action=[0,1]) - 1])
            next_state, reward, done = env.step(action=action, is_simulation=True, simulation_state=curr_state)
            if input_obs is None:
                next_state_obs = next_state if env.deterministic_transition else env.transition_distribution(state=curr_state, action=action)
                add_obs(obs=obs, unique_stat=unique_stat, state0=curr_state, action=action, state1=next_state_obs, reward=reward, done=done)
            curr_state = next_state
            reward_acc.append(reward)

            if env.deterministic_transition is True and next_state in history: #loop exit
                history.append(next_state)
                break
            history.append(next_state)
        state_history.append(history)
        reward_history.append(reward_acc)
        samples_history.append(all_q_samples)

    if input_obs is None:
        if output_obs is True:
            obs_history.append(deepcopy(obs))

        fitted_diag_std = torch.tensor(np.std(all_q_samples, axis=0, ddof=1))
        q_sample, step_size, all_q_samples, acc_prob = sample_q_mcmc(obs=obs, env=env, epsilon=epsilon, sigma=sigma, num_samples=num_samples, 
                                                                step_size=step_size, num_steps=num_steps, num_warmup_runs=num_warmup_runs,
                                                                num_warmup_samples_per_run=num_warmup_samples_per_run, fitted_diag_std=fitted_diag_std, 
                                                                target_acc_prob=target_acc_prob, step_size_rates=step_size_rates, 
                                                                disable_progbar=disable_progbar)
        samples_history.append(all_q_samples)
        acc_prob_history.append(acc_prob)
    
    if save_path is not None:
        save_dict = {
            "state_history": state_history,
            "reward_history": reward_history,
            "samples_history": samples_history,
            "env_data": (env.action_map, env.deterministic_transition),
            "acc_prob_history": acc_prob_history

        }
        if output_obs:
            save_dict["obs_history"] = obs_history
        
        np.save(save_path, save_dict)

    if output_obs:
        return state_history, reward_history, samples_history, acc_prob_history, obs_history
    else:
        return state_history, reward_history, samples_history, acc_prob_history

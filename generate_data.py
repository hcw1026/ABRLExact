import os
import datetime
import argparse
from functools import partial
from tqdm.notebook import tqdm

import multiprocessing
from omegaconf import OmegaConf

from ABRLExact.environment import DeepSea, DeepSeaPyramid, DeepSeaSwirl
from ABRLExact.CDFAgent import run_exact_deepsea

def worker_map(args): # pylint: disable=redefined-outer-name
    _, env_generator, kwargs = args # pylint: disable=redefined-outer-name
    if callable(env_generator):
        env = env_generator()
    else:
        env = env_generator
    return run_exact_deepsea(env, **kwargs)


def run_parallel_exact_deepsea(num_experiments, env_generator, epsilon=0.02, sigma=10, num_episodes=30, use_qmc=False, qmc_sobol_power=18, # pylint: disable=redefined-outer-name
                               output_obs=False, batch_size=1024, save_dir=None, n_jobs=None): # pylint: disable=redefined-outer-name

    if n_jobs is None:
        try:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        except NotImplementedError:
            n_jobs = 1

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    base_kwargs = {
        "epsilon": epsilon,
        "sigma": sigma,
        "num_episodes": num_episodes,
        "use_qmc": use_qmc,
        "qmc_sobol_power": qmc_sobol_power,
        "output_obs": output_obs,
        "disable_tqdm": True,
        "batch_size": batch_size
    }

    tasks = []
    for i in range(num_experiments):
        kwargs = base_kwargs.copy()
        if save_dir is not None:
            kwargs["save_path"] = os.path.join(save_dir, f"run_{i}.npy")

        tasks.append((i, env_generator, kwargs))

    print(f"Starting {num_experiments} experiments on {n_jobs} cores...")

    with multiprocessing.Pool(processes=n_jobs) as pool:
        output = list(tqdm(pool.imap(worker_map, tasks), total=num_experiments, desc="Parallel Experiments"))

    return output


def make_Deepsea_env(depth=5, deterministic_transition=True, randomised_actions=False, randomised_action_seed=None, penalty=None, sto_trans_prob=None):
    return DeepSea(depth=depth, 
                   starting_state=(0,0), 
                   goal_state=(-1,-1), 
                   deterministic_transition=deterministic_transition, 
                   randomised_actions=randomised_actions, 
                   randomised_action_seed=randomised_action_seed, 
                   penalty=penalty,
                   sto_trans_prob=sto_trans_prob)
    

def make_DeepSeaPyramid_env(depth=5, deterministic_transition=True, randomised_actions=False, randomised_action_seed=None, sto_trans_prob=None):
    return DeepSeaPyramid(depth=depth, 
                   starting_state=(0,0), 
                   goal_state=(-1,-1), 
                   deterministic_transition=deterministic_transition, 
                   randomised_actions=randomised_actions, 
                   randomised_action_seed=randomised_action_seed, 
                   sto_trans_prob=sto_trans_prob)
    

def make_DeepSeaSwirl_env(depth=5, deterministic_transition=True, randomised_actions=False, randomised_action_seed=None, penalty=None, sto_trans_prob=None):
    return DeepSeaSwirl(depth=depth, 
                   starting_state=(0,0), 
                   goal_state=(-1,-1), 
                   deterministic_transition=deterministic_transition, 
                   randomised_actions=randomised_actions, 
                   randomised_action_seed=randomised_action_seed, 
                   penalty=penalty,
                   sto_trans_prob=sto_trans_prob)


def get_run_name():
    return datetime.datetime.now().strftime("experiment_%Y%m%d_%H%M%S")


def find_existing_run(base_dir, config): # pylint: disable=redefined-outer-name
    if not os.path.exists(base_dir):
        return None
    
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            config_path = os.path.join(entry.path, "config.yaml")
            if os.path.exists(config_path):
                existing_config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
                if existing_config == config:
                    return entry.path
    return None


def configure_experiment(omega_config): # pylint: disable=redefined-outer-name
    config = OmegaConf.to_container(omega_config, resolve=True) # pylint: disable=redefined-outer-name

    env_name = config["name"].lower()

    if env_name == "deepsea":
        env_class = DeepSea
        use_penalty = True
    elif env_name == "deepseapyramid":
        env_class = DeepSeaPyramid
        use_penalty = False
    elif env_name == "deepseaswirl":
        env_class = DeepSeaSwirl
        use_penalty = True
    else:
        raise ValueError(f"Unknown environment name: {config['name']}")

    env_kwargs = {k: config[k] for k in ["depth", "deterministic_transition", "randomised_actions", "randomised_action_seed", "sto_trans_prob"]}
    if use_penalty:
        env_kwargs["penalty"] = config["penalty"]

    env_generator = partial(env_class, **env_kwargs) # pylint: disable=redefined-outer-name
                            
    return config, env_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel DeepSea experiments.")
    parser.add_argument("config", type=str, nargs="?", default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--base_dir", type=str, default="./results", help="Base directory for results.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs (cores) to use.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for exact solution computation.")
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon value from config.")
    
    args = parser.parse_args()
    
    base_dir = args.base_dir

    if not os.path.exists(args.config):
        print(f"Configuration file '{args.config}' not found.")
        exit(1)

    omega_config = OmegaConf.load(args.config)

    if args.epsilon is not None:
        omega_config.epsilon = args.epsilon
    
    config, env_generator = configure_experiment(omega_config)
    existing_run = find_existing_run(base_dir, config)

    if existing_run:
        print(f"Found existing run with same configuration at: {existing_run}")
        print("Skipping...")
    else:
        run_name = get_run_name()
        save_dir = os.path.join(base_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)
        
        OmegaConf.save(config=OmegaConf.create(config), f=os.path.join(save_dir, "config.yaml"))
        print(f"Created new run directory: {save_dir}")

        results = run_parallel_exact_deepsea(
            num_experiments=config["num_experiments"],
            env_generator=env_generator, 
            epsilon=config["epsilon"],
            sigma=config["sigma"],
            num_episodes=config["num_episodes"],
            use_qmc=config["use_qmc"],
            qmc_sobol_power=config["qmc_sobol_power"],
            output_obs=config["output_obs"],
            batch_size=args.batch_size,
            save_dir=save_dir,
            n_jobs=args.n_jobs
        )

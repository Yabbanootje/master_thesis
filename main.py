import omnisafe
import torch
import shutil
import os
import random as rand
import re
from multiprocessing.pool import Pool
from itertools import product
from omnisafe.utils.config import get_default_kwargs_yaml
from custom_envs.hand_made_levels.hm_curriculum_env import HMCurriculumEnv
from custom_envs.hand_made_levels.hm_adaptive_curriculum_env import HMAdaptiveCurriculumEnv
import warnings

def get_configs(folder, algorithms, epochs, cost_limit, seed, save_freq = None, steps_per_epoch = 1000, 
                update_iters = 1, nn_size = 256, lag_multiplier_init = 0.1, lag_multiplier_lr = 0.01,
                beta = 1.0, kappa = 20, early_stop_before = 0):
    """
    Function to get a list of configs for each of the given algorithms.

    Parameters:
        folder (str): the relative folder path.
        algorithms ([str]): list of the algorithm names from Omnisafe to create config for.
        epochs (int): amount of epochs agent should be trained for.
        cost_limit (int): the cost_limit to be used in the environments.
        seed (int): the seed for training the agents.
        save_freq (int | None, optional): amount of epochs between saves of the model. If None, only at the start and end of training.
        steps_per_epoch (int, optional): the number of steps before the policy is updated.
        update_iters (int), optional: the number of update iterations per update.
        nn_size (int, optional): the amount of neurons in given to the first and second hidden layers.
        lag_multiplier_init (float, optional): initial value of the Lagrangian multiplier if applicable.
        lag_multiplier_lr (float, optional): learning rate of the Lagrangian multiplier if applicable.
        beta (float, optional): fraction of the cost_limit that should be satisfied for kappa epochs, as described in the thesis. 
            Only applies to adaptive curricula.
        kappa (int, optional): amount of epochs in which the agent should be below beta x cost_limit, as described in the thesis.
            Only applies to adaptive curricula.
        early_stop_before (int, optional): the first task in the sequence where we do not want to stop training after the agent is ready for
            the next task. Is used to speed up training, if it is not necessary to train earlier tasks to the last epoch and 
            instead it is preferable to immediately start training the agent on the next task in the sequence.
            Only applies to adaptive curricula.
    """

    # If save_freq is not given only save at the start and end
    if save_freq == None:
        save_freq = epochs

    # Use cuda if available
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Configs to be returned
    custom_cfgs = []

    # Create a config for each algorithm
    for algo in algorithms:
        # Retrieve the available configs
        kwargs = get_default_kwargs_yaml(algo, None, "on-policy").todict()

        # Generic custom configs
        custom_cfg = {
            'seed': seed,
            'train_cfgs': {
                'device': device,
                'total_steps': epochs * steps_per_epoch,
            },
            'algo_cfgs': {
                'steps_per_epoch': steps_per_epoch,
                'update_iters': update_iters,
            },
            'logger_cfgs': {
                'log_dir': f"./results/" + folder,
                'save_model_freq': save_freq,
                'use_wandb': False,
                'wandb_project': folder.split("/")[0],
            },
            'model_cfgs': {
                'actor': {
                    'hidden_sizes': [nn_size, nn_size]
                },
                'critic': {
                    'hidden_sizes': [nn_size, nn_size]
                },
            },
        }

        # If the agents are trained using an adaptive curriculum, add adaptive configs
        if "adaptive_curriculum" == os.path.basename(folder):
            env_cfgs = {
                'beta': beta,
                'kappa': kappa,
                'early_stop_before': early_stop_before,
            }
            custom_cfg["env_cfgs"] = env_cfgs

        # Add cost_limit depending on specific algorithm and Lagrangian parameters if applicable
        if kwargs.get("lagrange_cfgs"):
            custom_cfg.update({'lagrange_cfgs': {
                'cost_limit': cost_limit,
                'lagrangian_multiplier_init': lag_multiplier_init,
            },
            })
            if kwargs["lagrange_cfgs"].get("lambda_lr"):
                custom_cfg['lagrange_cfgs'].update({'lambda_lr': lag_multiplier_lr,})
        if kwargs.get("algo_cfgs"):
            if kwargs["algo_cfgs"].get("cost_limit"):
                custom_cfg["algo_cfgs"].update({'cost_limit': cost_limit,})

        print(f"Initializing {algo}: {custom_cfg}")

        custom_cfgs.append(custom_cfg)

    return custom_cfgs

def get_agents(folder, algorithms, env_id, cfgs, curr_changes):
    """
    Function to get a list of agents using the given algorithms and configs.

    Parameters:
        folder (str): the relative folder path.
        algorithms ([str]): list of the algorithm names from Omnisafe to create config for. Should have the same length as cfgs.
        env_id (str): id by which the environment will be created. For example, 'SafetyPointFrom0HMT', means that a static
            curriculum will be created starting from task 0 all the way up to task T using the SafetyPoint environments from
            Omnisafe.
        cfgs ([dict]): list of configs to be used by the agents in the environment. Should have the same length as algorithms.
        curr_changes ([int]): the epochs at which a task change to the next task should occur, needs to be a strictly increasing 
            list. Only applicable to static curricula.
    """
    # Agents to be returned
    agents = []

    # Give each algorithm its respective configs
    for algorithm, cfg in zip(algorithms, cfgs):
        # Create Omnisafe agent using the configs
        agent = omnisafe.Agent(algorithm, env_id, custom_cfgs=cfg)

        # Check whether a curriculum is necessary
        if "From" in env_id:
            # Get the start task
            start_version_pattern = r'From(\d+|T)'
            start_version = re.search(start_version_pattern, env_id)
            start_task = start_version.group(1)
            if start_task == "T":
                start_task = len(curr_changes)
            
            if int(start_task) != 0:
                # If start task is not the first, load the model that was trained on the task before the start task
                algo_folders = os.listdir(f"results/" + folder)
                algo_folder = [fldr for fldr in algo_folders if algorithm in fldr and any(prefix + str(int(start_task) - 1) in fldr for prefix in ["HM", "HMR", "HMA"])][0]
                algo_path = os.path.join(f"results/", folder, algo_folder)
                seed_folder = [fldr for fldr in os.listdir(algo_path) if "seed-" + str(cfg.get("seed")).zfill(3) in fldr][0]
                if "HMA" in env_id:
                    agent.agent.load(path=os.path.join(algo_path, seed_folder))
                else:
                    agent.agent.load(epoch=curr_changes[int(start_task) - 1], path=os.path.join(algo_path, seed_folder))

        agents.append(agent)

    return agents

def train_agent(agent, episodes = 1, render_episodes = 1, epochs_to_render = []):
    """
    Function to train the given agent with the given evaluation parameters.

    Parameters:
        agent (omnisafe.Agent): agent to be trained.
        episodes (int, optional): amount of evaluation episodes.
        render_episodes (int, optional): amount of evaluation episodes that should be rendered, which is done on new episodes.
        epochs_to_render ([int], optional): specific epochs that should be rendered instead of all epochs that have a saved model.
    """
    # Train the agent
    agent.learn()

    # Plot generic progress of the agent
    agent.plot(smooth=1)

    # Evaluate the agent without rendered episodes
    if episodes >= 1:
        agent.evaluate(num_episodes=episodes)

    # Evaluate the agent with rendered episodes
    if render_episodes >= 1:
        agent.render(num_episodes=render_episodes, render_mode='rgb_array', width=256, height=256, 
                     epochs_to_render=epochs_to_render)

def run_experiment(folder_base, cost_limit, seed, save_freq, epochs, algorithm, algorithm_type, curr_changes, eval_episodes, 
                   render_episodes, end_task, start_task = 0, beta = 1.0, kappa = 20, early_stop_before = 0):
    """
    Function to run experiments with the given parameters.

    Parameters:
        folder_base (str): the name of the folder where all experiments should be saved.
        cost_limit (int): the cost_limit to be used in the environments.
        seed (int): the seed for training the agents.
        save_freq (int | None): amount of epochs between saves of the model. If None, only at the start and end of training.
        epochs (int): amount of epochs agent should be trained for.
        algorithm (str): the algorithm name from Omnisafe to train.
        algorithm_type (str): the type of curriculum to be used with the algorithm.
        curr_changes ([int]): the epochs at which a task change to the next task should occur, needs to be a strictly increasing 
            list. Only applicable to static curricula.
        eval_episodes (int): amount of evaluation episodes.
        render_episodes (int): amount of evaluation episodes that should be rendered, which is done on new episodes.
        end_task (int): task to start training from. Only applicable to static curricula.
        start_task (int, optional): task to start training from. Only applicable to static curricula.
        beta (float, optional): fraction of the cost_limit that should be satisfied for kappa epochs, as described in the thesis. 
            Only applicable to adaptive curricula.
        kappa (int, optional): amount of epochs in which the agent should be below beta x cost_limit, as described in the thesis.
            Only applicable to adaptive curricula.
        early_stop_before (int, optional): the first task in the sequence where we do not want to stop training after the agent is ready for
            the next task. Is used to speed up training, if it is not necessary to train earlier tasks to the last epoch and 
            instead it is preferable to immediately start training the agent on the next task in the sequence.
            Only applicable to adaptive curricula.
    """
    
    # Create the appropriate env_id depending on the type of curriculum
    if algorithm_type == "baseline":
        env_id = f'SafetyPointHM{end_task if end_task < 6 else "T"}-v0'
    elif algorithm_type == "curriculum":
        env_id = f'SafetyPointFrom{start_task if end_task < 6 else "T"}HM{end_task if end_task < 6 else "T"}-v0'
    elif algorithm_type == "adaptive_curriculum":
        env_id = f'SafetyPointFrom{end_task if end_task < 6 else "T"}HMA{end_task if end_task < 6 else "T"}-v0'
    else:
        raise Exception("Invalid algorithm type, must be 'baseline', 'curriculum' or 'adaptive_curriculum'.")
    
    # Use a seperate folder for each type
    folder=f"{folder_base}/{algorithm_type}"

    # Get configurations
    cfgs = get_configs(folder=folder, algos=[algorithm], epochs=epochs, cost_limit=cost_limit, seed=seed, 
                       save_freq = save_freq, beta=beta, kappa=kappa, early_stop_before=early_stop_before)

    # Initialize agents
    agents = get_agents(folder, [algorithm], env_id, cfgs, curr_changes)

    # Train agents
    for agent in agents:
        train_agent(agent, eval_episodes, render_episodes, True, [int(epochs/4), int(epochs/2), int(3 * epochs/4), epochs])


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 3000
    repetitions = 10
    baseline_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
    curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    adapt_curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    folder_base = "incremental_adaptive_curriculum"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [5905, 7337, 572, 5689, 3968, 175, 4678, 9733, 3743, 7596] # [int(rand.random() * 10000) for i in range(repetitions)]
    betas = [0.5, 1.0, 1.5]
    kappas = [5, 10, 20]
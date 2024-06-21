import omnisafe
import torch
import shutil
import os
import argparse
import random as rand
import re
from multiprocessing.pool import Pool
import wandb
from itertools import product
from omnisafe.utils.config import get_default_kwargs_yaml
from custom_envs.hand_made_levels.hm_curriculum_env import HMCurriculumEnv
from plot_functions import *

def get_configs(folder, algos, epochs, cost_limit, seed, save_freq = None, steps_per_epoch = 1000, 
                update_iters = 1, nn_size = 256, lag_multiplier_init = 0.1, lag_multiplier_lr = 0.01,
                focops_eta = 0.02, focops_lam = 1.5):
    """
    steps_per_epoch (int): the number of steps before the policy is updated
    update_iters (int): the number of update iterations per update
    """

    if save_freq == None:
        save_freq = epochs

    if torch.cuda.is_available():
        device = "cuda:0"
        use_wandb = True
    else:
        device = "cpu"
        use_wandb = False

    custom_cfgs = []

    for algo in algos:
        # cfg_path = get_yaml_path(algo, "on-policy")
        # kwargs = load_yaml(cfg_path)
        kwargs = get_default_kwargs_yaml(algo, None, "on-policy").todict()

        custom_cfg = {
            'seed': seed,
            'train_cfgs': {
                'device': device,
                'total_steps': epochs * steps_per_epoch,
            },
            'algo_cfgs': {
                'steps_per_epoch': steps_per_epoch,
                'update_iters': update_iters,
                # 'penalty_coef': 0.05,
            },
            'logger_cfgs': {
                'log_dir': "./app/results/" + folder,
                'save_model_freq': save_freq,
                'use_wandb': use_wandb,
                'wandb_project': folder.split("/")[0],
            },
            'model_cfgs': {
                'actor': {
                    'hidden_sizes': [nn_size, nn_size]
                },
                'critic': {
                    'hidden_sizes': [nn_size, nn_size]
                },
                # 'linear_lr_decay': False,
                # 'std_range': [0.1, 0.0]
            }
        }

        # Add cost_limit depending on specific algorithm, the cost is discounted
        if kwargs.get("lagrange_cfgs"):
            custom_cfg.update({'lagrange_cfgs': {
                'cost_limit': cost_limit,
                'lagrangian_multiplier_init': lag_multiplier_init,
            },
            })
            if kwargs["lagrange_cfgs"].get("lambda_lr"):
                custom_cfg['lagrange_cfgs'].update({'lambda_lr': lag_multiplier_lr,})
        if kwargs.get("algo_cfgs"):
            if kwargs["algo_cfgs"].get("focops_eta"):
                custom_cfg["algo_cfgs"].update({'focops_eta': focops_eta,})
            if kwargs["algo_cfgs"].get("focops_lam"):
                custom_cfg["algo_cfgs"].update({'focops_lam': focops_lam,})
            if kwargs["algo_cfgs"].get("cost_limit"):
                custom_cfg["algo_cfgs"].update({'cost_limit': cost_limit,})

        print(f"{algo}: {custom_cfg}")

        custom_cfgs.append(custom_cfg)

    return custom_cfgs

def get_agents(folder, algorithms, env_id, cfgs, curr_changes):
    agents = []
    for algorithm, cfg in zip(algorithms, cfgs):
        agent = omnisafe.Agent(algorithm, env_id, custom_cfgs=cfg)
        if "From" in env_id:
            start_version_pattern = r'From(\d+|T)'
            start_version = re.search(start_version_pattern, env_id)
            start_task = start_version.group(1)
            
            if int(start_task) != 0:
                algo_folders = os.listdir("app/results/" + folder)
                algo_folder = [fldr for fldr in algo_folders if algorithm in fldr and "HM" + start_task in fldr][0]
                algo_path = os.path.join("app/results/", folder, algo_folder)
                seed_folder = [fldr for fldr in os.listdir(algo_path) if "seed-" + str(cfg.get("seed")).zfill(3) in fldr][0]
                agent.agent.load(curr_changes[int(start_task) - 1], os.path.join(algo_path, seed_folder))
        agents.append(agent)

    return agents

def train_agent(agent, episodes = 1, render_episodes = 1, make_videos = False, epochs_to_render = []):
    agent.learn()

    agent.plot(smooth=1)

    if episodes >= 1:
        agent.evaluate(num_episodes=episodes)

    if make_videos and render_episodes >= 1:
        agent.render(num_episodes=render_episodes, render_mode='rgb_array', width=256, height=256, 
                     epochs_to_render=epochs_to_render)
        
    # Remove wandb folder
    wandb_path = os.path.join(agent.agent.logger._log_dir, "wandb")
    shutil.rmtree(wandb_path, ignore_errors=True)

def remove_wandb(folder):
    for type in ["baseline", "curriculum"]:
        algo_folders = os.listdir("app/results/" + folder + "/" + type)
        for algo_folder in algo_folders:
            algo_path = os.path.join("app/results/", folder, type, algo_folder)
            seed_folders = os.listdir(algo_path)
            for seed_folder in seed_folders:
                wandb_path = os.path.join("app/results/", folder, type, algo_folder, seed_folder, "wandb")
                print(wandb_path)
                shutil.rmtree(wandb_path, ignore_errors = True)

def run_experiment(eval_episodes, render_episodes, cost_limit, seed, save_freq, epochs, algorithm, env_id, folder, curr_changes):
    # Get configurations
    cfgs = get_configs(folder=folder, algos=[algorithm], epochs=epochs, cost_limit=cost_limit, seed=seed, 
                       save_freq = save_freq)

    # Initialize agents
    agents = get_agents(folder, [algorithm], env_id, cfgs, curr_changes)

    # Train agents
    for agent in agents:
        train_agent(agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

def use_params(algorithm, end_task, algorithm_type, seed):
    if algorithm_type == "baseline":
        env_id = f'SafetyPointHM{end_task if end_task < 6 else "T"}-v0'
    elif algorithm_type == "curriculum":
        env_id = f'SafetyPointFrom{end_task - 1}HMR{end_task if end_task < 6 else "T"}-v0'
    else:
        raise Exception("Invalid algorithm type, must be either 'baseline' or 'curriculum'.")

    run_experiment(eval_episodes=eval_episodes, render_episodes=render_episodes, cost_limit=cost_limit, 
                    seed=seed, save_freq=save_freq, epochs=epochs, algorithm=algorithm, 
                    env_id=env_id, folder=folder_base + "/" + algorithm_type, curr_changes=curr_changes)

if __name__ == '__main__':
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 2000
    repetitions = 10
    baseline_algorithms = ["PPOLag"]#, "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
    curr_algorithms = ["PPOLag"]#, "FOCOPS", "CUP", "PPOEarlyTerminated"]
    folder_base = "algorithm_comparison"
    curr_changes = [10, 20, 40, 100]#, 300, 700]
    seeds = [7337, 175, 4678, 9733, 3743, 572, 5689, 3968, 7596, 5905] # [int(rand.random() * 10000) for i in range(repetitions)]

    # Repeat experiments
    # wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
    # for end_task in range(1, len(curr_changes) + 1):
    #     with Pool(8) as p:
    #         args_base = list(product(baseline_algorithms, [end_task], ["baseline"], seeds))
    #         args_curr = list(product(curr_algorithms, [end_task], ["curriculum"], seeds))
    #         args = args_curr + args_base
    #         p.starmap(use_params, args)

    # for i in range(7):
    #     use_params(*("PPOLag", i, "baseline", 42))

    # Plot the results
    # train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
    # eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, save_freq=save_freq)
    # print_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)
    
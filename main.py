import omnisafe
import torch
import os
import random as rand
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
# from safety_gymnasium.utils.registration import register
# from custom_envs.preliminary_levels.curriculum_env import CurriculumEnv
# from omnisafe.utils.tools import get_yaml_path, load_yaml
from omnisafe.utils.config import get_default_kwargs_yaml
from custom_envs.hand_made_levels.hm_curriculum_env import HMCurriculumEnv

def get_configs(folder, algos, epochs, cost_limit, random):
    steps_per_epoch = 1000
    safe_freq = 30
    
    if random:
        seed = int(rand.random() * 10000)
    else:
        seed = 0

    custom_cfgs = []

    for algo in algos:
        # cfg_path = get_yaml_path(algo, "on-policy")
        # kwargs = load_yaml(cfg_path)
        kwargs = get_default_kwargs_yaml(algo, None, "on-policy").todict()

        custom_cfg = {
            'seed': seed,
            'train_cfgs': {
                'total_steps': epochs * steps_per_epoch,
                'vector_env_nums': 1,
                'parallel': 1,
            },
            'algo_cfgs': {
                'steps_per_epoch': steps_per_epoch,
                'update_iters': 1,
                # 'penalty_coef': 0.01,
            },
            'logger_cfgs': {
                'log_dir': "./app/results/" + folder,
                'save_model_freq': safe_freq,
                # 'use_wandb': True,
                # 'wandb_project': "TODO",
            },
            'model_cfgs': {
                'actor': {
                    'hidden_sizes': [255, 255]
                },
                'critic': {
                    'hidden_sizes': [255, 255]
                },
                # 'std_range': [0.1, 0.0]
            }
        }

        # Add cost_limit depending on specific algorithm
        if kwargs.get("lagrange_cfgs"):
            custom_cfg.update({'lagrange_cfgs': {
                'cost_limit': cost_limit,
                # 'lagrangian_multiplier_init': 0.1,
            },
            })
            # if kwargs["lagrange_cfgs"].get("lambda_lr"):
            #     custom_cfg['lagrange_cfgs'].update({'lambda_lr': 0.05,})
        if kwargs["algo_cfgs"].get("cost_limit"):
            custom_cfg["algo_cfgs"].update({'cost_limit': cost_limit,})

        print(f"{algo}: {custom_cfg}")

        custom_cfgs.append(custom_cfg)

    return custom_cfgs

def get_agents(algorithms, env_id, cfgs):
    agents = []
    for algorithm, cfg in zip(algorithms, cfgs):
        agents.append(omnisafe.Agent(algorithm, env_id, custom_cfgs=cfg))

    return agents

def train_agent(agent, episodes = 1, make_videos = False):
    agent.learn()

    agent.plot(smooth=1)

    agent.evaluate(num_episodes=episodes)

    if make_videos:
        agent.render(num_episodes=episodes, render_mode='rgb_array', width=256, height=256)

def plot_train(folder, curr_changes, cost_limit, include_weak=False):
    # Get folder names for all algorithms
    baseline_dir = "app/results/" + folder + "/baseline"
    curr_dir = "app/results/" + folder + "/curriculum"

    baseline_algorithms = [entry.name for entry in os.scandir(baseline_dir)]
    curr_algorithms = [entry.name for entry in os.scandir(curr_dir)]

    baseline_dfs = {}
    curr_dfs = {}

    for algorithm in baseline_algorithms:
        paths = [entry.path for entry in os.scandir(os.path.join(baseline_dir, algorithm))]
        df_list = [pd.read_csv(os.path.join(path, "progress.csv")) for path in paths]
        baseline_dfs[algorithm] = pd.concat(df_list)

    for algorithm in curr_algorithms:
        paths = [entry.path for entry in os.scandir(os.path.join(curr_dir, algorithm))]
        df_list = [pd.read_csv(os.path.join(path, "progress.csv")) for path in paths]
        curr_dfs[algorithm] = pd.concat(df_list)

    baseline_rewards_mean = {}
    baseline_costs_mean = {}
    baseline_rewards_std = {}
    baseline_costs_std = {}

    for algorithm, df in baseline_dfs.items():
        baseline_rewards_mean[algorithm] = df.groupby(df.index)['Metrics/EpRet'].mean().values
        baseline_costs_mean[algorithm] = df.groupby(df.index)['Metrics/EpCost'].mean().values
        baseline_rewards_std[algorithm] = df.groupby(df.index)['Metrics/EpRet'].std().values
        baseline_costs_std[algorithm] = df.groupby(df.index)['Metrics/EpCost'].std().values

    curr_rewards_mean = {}
    curr_costs_mean = {}
    curr_rewards_std = {}
    curr_costs_std = {}

    for algorithm, df in curr_dfs.items():
        curr_rewards_mean[algorithm] = df.groupby(df.index)['Metrics/EpRet'].mean().values
        curr_costs_mean[algorithm] = df.groupby(df.index)['Metrics/EpCost'].mean().values
        curr_rewards_std[algorithm] = df.groupby(df.index)['Metrics/EpRet'].std().values
        curr_costs_std[algorithm] = df.groupby(df.index)['Metrics/EpCost'].std().values

    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)
        
    last_change = curr_changes[-1]

    plt.figure(figsize=(10, 5), dpi=80)

    # Plot baseline rewards
    for algorithm_name in baseline_rewards_mean.keys():
        plt.plot(np.arange(1, len(baseline_rewards_mean[algorithm_name]) + 1), baseline_rewards_mean[algorithm_name], label="Baseline - " + algorithm_name.split('-')[0])
        plt.fill_between(x=np.arange(1, len(baseline_rewards_mean[algorithm_name]) + 1),
                        y1=baseline_rewards_mean[algorithm_name] - baseline_rewards_std[algorithm_name],
                        y2=baseline_rewards_mean[algorithm_name] + baseline_rewards_std[algorithm_name], alpha=0.2)

    # Plot curriculum rewards
    for algorithm_name in curr_rewards_mean.keys():
        plt.plot(np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1), curr_rewards_mean[algorithm_name], label="Curriculum Strong - " + algorithm_name.split('-')[0])
        plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1),
                        y1=curr_rewards_mean[algorithm_name] - curr_rewards_std[algorithm_name],
                        y2=curr_rewards_mean[algorithm_name] + curr_rewards_std[algorithm_name], alpha=0.2)

        if include_weak:
            plt.plot(np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1 - last_change), curr_rewards_mean[algorithm_name][last_change:], label="Curriculum Weak - " + algorithm_name.split('-')[0])
            plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1 - last_change),
                            y1=(curr_rewards_mean[algorithm_name] - curr_rewards_std[algorithm_name])[last_change:],
                            y2=(curr_rewards_mean[algorithm_name] + curr_rewards_std[algorithm_name])[last_change:], alpha=0.2)

    for change in curr_changes:
        plt.axvline(x=change, color="gray", linestyle='-')

    plt.legend(loc=(1.01, 0.01), ncol = 1)
    plt.tight_layout(pad = 2)
    plt.grid()
    plt.title("Rewards of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Reward")
    plt.savefig("app/figures/" + folder + "/rewards.png")
    plt.show()

    plt.figure(figsize=(10, 5), dpi=80)

    # Plot baseline costs
    for algorithm_name in baseline_costs_mean.keys():
        plt.plot(np.arange(1, len(baseline_costs_mean[algorithm_name]) + 1), baseline_costs_mean[algorithm_name], label="Baseline - " + algorithm_name.split('-')[0])
        plt.fill_between(x=np.arange(1, len(baseline_costs_mean[algorithm_name]) + 1),
                        y1=baseline_costs_mean[algorithm_name] - baseline_costs_std[algorithm_name],
                        y2=baseline_costs_mean[algorithm_name] + baseline_costs_std[algorithm_name], alpha=0.2)

    # Plot curriculum costs
    for algorithm_name in curr_costs_mean.keys():
        plt.plot(np.arange(1, len(curr_costs_mean[algorithm_name]) + 1), curr_costs_mean[algorithm_name], label="Curriculum Strong - " + algorithm_name.split('-')[0])
        plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1),
                        y1=curr_costs_mean[algorithm_name] - curr_costs_std[algorithm_name],
                        y2=curr_costs_mean[algorithm_name] + curr_costs_std[algorithm_name], alpha=0.2)

        if include_weak:
            plt.plot(np.arange(1, len(curr_costs_mean[algorithm_name]) + 1 - last_change), curr_costs_mean[algorithm_name][last_change:], label="Curriculum Weak - " + algorithm_name.split('-')[0])
            plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1 - last_change),
                            y1=(curr_costs_mean[algorithm_name] - curr_costs_std[algorithm_name])[last_change:],
                            y2=(curr_costs_mean[algorithm_name] + curr_costs_std[algorithm_name])[last_change:], alpha=0.2)

    plt.axhline(y=cost_limit, color='r', linestyle='-')

    for change in curr_changes:
        plt.axvline(x=change, color="gray", linestyle='-')

    plt.legend(loc=(1.01, 0.01), ncol = 1)
    plt.tight_layout(pad = 2)
    plt.grid()
    plt.title("Costs of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Cost")
    plt.savefig("app/figures/" + folder + "/costs.png")
    plt.show()

def plot_eval(folder, curr_changes, cost_limit, include_weak=False):
    def extract_values(pattern, text):
        return [float(match.group(1)) for match in re.finditer(pattern, text)]

    def process_data(algorithms, directory, rewards_mean, costs_mean, lengths_mean, rewards_std, costs_std, lengths_std, indices):
        for algorithm in algorithms:
            seed_paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            video_paths = [os.path.join(path, "video") for path in seed_paths]

            epoch_data = {}

            for path in video_paths:
                epochs = [entry.name for entry in os.scandir(path)]

                for epoch in epochs:
                    with open(os.path.join(path, epoch, "result.txt"), 'r') as file:
                        data = file.read()

                        rewards = extract_values(r'Episode reward: ([\d\.-]+)', data)
                        costs = extract_values(r'Episode cost: ([\d\.-]+)', data)
                        lengths = extract_values(r'Episode length: ([\d\.-]+)', data)

                        index = int(epoch.split("-")[1])
                        if index not in epoch_data:
                            epoch_data[index] = {'rewards': [], 'costs': [], 'lengths': []}
                        epoch_data[index]['rewards'].append(rewards)
                        epoch_data[index]['costs'].append(costs)
                        epoch_data[index]['lengths'].append(lengths)

            for epoch_number, data in epoch_data.items():
                rewards_mean[algorithm] = rewards_mean.get(algorithm, []) + [np.mean(data['rewards'])]
                costs_mean[algorithm] = costs_mean.get(algorithm, []) + [np.mean(data['costs'])]
                lengths_mean[algorithm] = lengths_mean.get(algorithm, []) + [np.mean(data['lengths'])]
                rewards_std[algorithm] = rewards_std.get(algorithm, []) + [np.std(data['rewards'])]
                costs_std[algorithm] = costs_std.get(algorithm, []) + [np.std(data['costs'])]
                lengths_std[algorithm] = lengths_std.get(algorithm, []) + [np.std(data['lengths'])]

            if len(indices) == 0:
                for key in epoch_data.keys():
                    indices.setdefault(key)

    baseline_dir = "app/results/" + folder + "/baseline"
    curr_dir = "app/results/" + folder + "/curriculum"

    baseline_algorithms = [entry.name for entry in os.scandir(baseline_dir)]
    curr_algorithms = [entry.name for entry in os.scandir(curr_dir)]

    baseline_rewards_mean, baseline_costs_mean, baseline_lengths_mean = {}, {}, {}
    baseline_rewards_std, baseline_costs_std, baseline_lengths_std = {}, {}, {}
    curr_rewards_mean, curr_costs_mean, curr_lengths_mean = {}, {}, {}
    curr_rewards_std, curr_costs_std, curr_lengths_std = {}, {}, {}
    indices = {}

    process_data(baseline_algorithms, baseline_dir, baseline_rewards_mean, baseline_costs_mean, baseline_lengths_mean, 
                 baseline_rewards_std, baseline_costs_std, baseline_lengths_std, indices)
    process_data(curr_algorithms, curr_dir, curr_rewards_mean, curr_costs_mean, curr_lengths_mean, curr_rewards_std, 
                 curr_costs_std, curr_lengths_std, indices)
    
    indices = indices.keys()

    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)

    for data_type in ['rewards', 'costs', 'lengths']:
        plt.figure(figsize=(10, 5), dpi=80)

        for algorithm_name in eval(f"baseline_{data_type}_mean").keys():
            sorted_values = sorted(zip(indices, eval(f"baseline_{data_type}_mean")[algorithm_name], eval(f"baseline_{data_type}_std")[algorithm_name]))
            sorted_indices, sorted_means, sorted_stds = zip(*sorted_values)
            plt.plot(sorted_indices, sorted_means, label=f"Baseline - {algorithm_name.split('-')[0]}")
            plt.fill_between(x=sorted_indices,
                            y1=np.asarray(sorted_means) - np.asarray(sorted_stds),
                            y2=np.asarray(sorted_means) + np.asarray(sorted_stds), alpha=0.2)

        for algorithm_name in eval(f"curr_{data_type}_mean").keys():
            sorted_values = sorted(zip(indices, eval(f"curr_{data_type}_mean")[algorithm_name], eval(f"curr_{data_type}_std")[algorithm_name]))
            sorted_indices, sorted_means, sorted_stds = zip(*sorted_values)
            plt.plot(sorted_indices, sorted_means, label=f"Curriculum Strong - {algorithm_name.split('-')[0]}")
            plt.fill_between(x=sorted_indices,
                            y1=np.asarray(sorted_means) - np.asarray(sorted_stds),
                            y2=np.asarray(sorted_means) + np.asarray(sorted_stds), alpha=0.2)

        if data_type == 'costs':
            plt.axhline(y=cost_limit, color='r', linestyle='-')

        for change in curr_changes:
            plt.axvline(x=change, color="gray", linestyle='-')

        plt.legend(loc=(1.01, 0.01), ncol=1)
        plt.tight_layout(pad=2)
        plt.grid()
        plt.title(f"{data_type.capitalize()} of agent using curriculum and baseline agent during evaluation")
        plt.xlabel("Epochs")
        plt.ylabel(f"{data_type.capitalize()[:-1] if data_type != 'rewards' else data_type.capitalize()}")
        plt.savefig(f"app/figures/{folder}/{data_type}_eval.png")
        plt.show()

if __name__ == '__main__':
    eval_episodes = 3
    cost_limit = 0.0
    epochs = 150
    repetitions = 3
    baseline_algorithms = ["PPO", "PPOLag", "P3O"]
    curr_algorithms = ["PPOEarlyTerminated", "PPOLag", "CPPOPID", "CPO", "IPO", "P3O"]

    # Create folder
    folder_name = "test-half_curriculum-multi_algos_no-fork_no-std_no-lag-par"
    # folder_name = folder_name + "---" + str(datetime.datetime.now()).replace(' ', '-')

    # Repeat experiments
    for i in range(repetitions):
        # Get configurations
        base_cfgs = get_configs(folder=folder_name + "/baseline", algos=baseline_algorithms, epochs=epochs, cost_limit=cost_limit, random=True)
        curr_cfgs = get_configs(folder=folder_name + "/curriculum", algos=curr_algorithms, epochs=epochs, cost_limit=cost_limit, random=True)

        # Initialize agents
        baseline_env_id = 'SafetyPointHM3-v0'
        curr_env_id = 'SafetyPointHM0-v0'

        baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
        curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

        # Train agents
        for baseline_agent in baseline_agents:
            train_agent(baseline_agent, eval_episodes, True)

        for curriculum_agent in curriculum_agents:
            train_agent(curriculum_agent, eval_episodes, True)

    # Plot the results
    curr_changes = [10, 20, 30]
    plot_train(folder_name, curr_changes, cost_limit, include_weak=False)

    plot_eval(folder_name, curr_changes, cost_limit, include_weak=False)
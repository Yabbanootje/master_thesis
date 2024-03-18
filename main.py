import omnisafe
import torch
import os
import argparse
import random as rand
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
from itertools import product
# from safety_gymnasium.utils.registration import register
# from custom_envs.preliminary_levels.curriculum_env import CurriculumEnv
# from omnisafe.utils.tools import get_yaml_path, load_yaml
from omnisafe.utils.config import get_default_kwargs_yaml
from custom_envs.hand_made_levels.hm_curriculum_env import HMCurriculumEnv

def get_configs(folder, algos, epochs, cost_limit, random, safe_freq = None, steps_per_epoch = 1000, 
                update_iters = 10, nn_size = 64, lag_multiplier_init = 0.001, lag_multiplier_lr = 0.035):
    """
    steps_per_epoch (int): the number of steps before the policy is updated
    update_iters (int): the number of update iterations per update
    """

    if safe_freq == None:
        safe_freq = epochs
    
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
            },
            'algo_cfgs': {
                'steps_per_epoch': steps_per_epoch,
                'update_iters': update_iters,
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
                    'hidden_sizes': [nn_size, nn_size]
                },
                'critic': {
                    'hidden_sizes': [nn_size, nn_size]
                },
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
    means = {"rewards": [], "costs": []}

    plt.figure(figsize=(10, 5), dpi=80)

    # Plot baseline rewards
    for algorithm_name in baseline_rewards_mean.keys():
        plt.plot(np.arange(1, len(baseline_rewards_mean[algorithm_name]) + 1), baseline_rewards_mean[algorithm_name], label="Baseline - " + algorithm_name.split('-')[0])
        plt.fill_between(x=np.arange(1, len(baseline_rewards_mean[algorithm_name]) + 1),
                        y1=baseline_rewards_mean[algorithm_name] - baseline_rewards_std[algorithm_name],
                        y2=baseline_rewards_mean[algorithm_name] + baseline_rewards_std[algorithm_name], alpha=0.2)

    # Plot curriculum rewards
    for algorithm_name in curr_rewards_mean.keys():
        if means['rewards'] == []:
            means['rewards'] = curr_rewards_mean[algorithm_name]
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
    plt.close()

    plt.figure(figsize=(10, 5), dpi=80)

    # Plot baseline costs
    for algorithm_name in baseline_costs_mean.keys():
        plt.plot(np.arange(1, len(baseline_costs_mean[algorithm_name]) + 1), baseline_costs_mean[algorithm_name], label="Baseline - " + algorithm_name.split('-')[0])
        plt.fill_between(x=np.arange(1, len(baseline_costs_mean[algorithm_name]) + 1),
                        y1=baseline_costs_mean[algorithm_name] - baseline_costs_std[algorithm_name],
                        y2=baseline_costs_mean[algorithm_name] + baseline_costs_std[algorithm_name], alpha=0.2)

    # Plot curriculum costs
    for algorithm_name in curr_costs_mean.keys():
        if means['costs'] == []:
            means['costs'] = curr_costs_mean[algorithm_name]
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
    plt.close()

    return means

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

    means = {"rewards": [], "costs": [], "lengths": []}

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
            if means[data_type] == []:
                means[data_type] = sorted_means
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
        plt.close()

    return means

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lag_multiplier_lr', dest='lag_multiplier_lr', type=float, help='Add lagrangian multiplier learning rate')
    args = parser.parse_args()

    eval_episodes = 3
    # cost_limit = 1.0
    epochs = 100
    repetitions = 3
    baseline_algorithms = ["PPOLag"] # ["PPO", "PPOLag", "P3O"]
    curr_algorithms = ["PPOLag"] # ["PPOEarlyTerminated", "PPOLag", "CPPOPID", "CPO", "IPO", "P3O"]

    # Grid search params
    cost_limits = [5.0] # [1.0, 5.0, 10.0]
    lag_multiplier_inits = [0.001, 0.01, 0.1] # [0.001, 0.005, 0.01, 0.1]
    lag_multiplier_lrs = [0.01, 0.035, 0.05, 0.1] # [args.lag_multiplier_lr]
    steps_per_epochs = [1000] # [500, 1000, 2000]
    update_iterss = [1, 10, 50]
    nn_sizes = [64, 256] # [64, 128, 256]

    last_means = pd.DataFrame(columns = ["lag_multiplier_inits", "lag_multiplier_lrs", "update_iterss", "nn_sizes", 
                                         "reward", "cost", "eval_reward", "eval_cost", "eval_length"]).set_index(
                                             ["lag_multiplier_inits", "lag_multiplier_lrs", "update_iterss", "nn_sizes"]
                                        )

    for grid_params in product(cost_limits, lag_multiplier_inits, lag_multiplier_lrs, steps_per_epochs, update_iterss, nn_sizes):
        (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size) = grid_params
        # Create folder
        folder_name = "grid_search/test-half_curriculum-" + str(grid_params)
        # folder_name = folder_name + "---" + str(datetime.datetime.now()).replace(' ', '-')

        # Repeat experiments
        # for i in range(repetitions):
        #     # Get configurations
        #     base_cfgs = get_configs(folder=folder_name + "/baseline", algos=baseline_algorithms, epochs=epochs, cost_limit=cost_limit, random=True,
        #                             steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size, safe_freq = 20,
        #                             lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)
        #     curr_cfgs = get_configs(folder=folder_name + "/curriculum", algos=curr_algorithms, epochs=epochs, cost_limit=cost_limit, random=True,
        #                             steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size, safe_freq = 20,
        #                             lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)

        #     # Initialize agents
        #     baseline_env_id = 'SafetyPointHM3-v0'
        #     curr_env_id = 'SafetyPointHM0-v0'

        #     baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
        #     curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

        #     # Train agents
        #     for baseline_agent in baseline_agents:
        #         train_agent(baseline_agent, eval_episodes, True)

        #     for curriculum_agent in curriculum_agents:
        #         train_agent(curriculum_agent, eval_episodes, True)

        # Plot the results
        curr_changes = [10, 20, 30]
        means = plot_train(folder_name, curr_changes, cost_limit, include_weak=False)
        eval_means = plot_eval(folder_name, curr_changes, cost_limit, include_weak=False)

        reward = means["rewards"][-1]
        cost = means["costs"][-1]
        eval_reward = eval_means["rewards"][-1]
        eval_cost = eval_means["costs"][-1]
        eval_length = eval_means["lengths"][-1]

        parameter_means = pd.DataFrame(data = {"lag_multiplier_inits": lag_multiplier_init, "lag_multiplier_lrs": lag_multiplier_lr, 
                                "update_iterss": update_iters, "nn_sizes": nn_size, 
                                "reward": reward, "cost": cost, 'eval_reward': eval_reward,'eval_cost': eval_cost, 
                                'eval_length': eval_length}, 
                                index = [0]).set_index(
                                    ["lag_multiplier_inits", "lag_multiplier_lrs", "update_iterss", "nn_sizes"]
                                )       

        last_means = pd.concat([last_means, parameter_means])

    last_means = last_means.sort_values(by=["eval_reward"])

    # Get annotation for heatmap
    annotation = last_means.to_numpy()

    # Normalize columns
    for column in last_means.columns:
        if column == "eval_cost" or column == "cost":
            last_means[column] = np.log(last_means[column] + 0.1)
        last_means[column] = (last_means[column] - last_means[column].min()) / (last_means[column].max() - last_means[column].min())

    # Plotting the heatmap
    plt.figure(figsize=(8, 12))
    plt.imshow(last_means.values, cmap='viridis', aspect='auto')

    # Add labels and ticks
    plt.title('Heatmap of final epoch performance')
    plt.ylabel('Parameter Combinations (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size)')
    plt.xlabel('Metrics')
    plt.yticks(ticks=np.arange(len(last_means.index)), labels=last_means.index, rotation='horizontal')
    plt.xticks(ticks=np.arange(len(last_means.columns)), labels=last_means.columns)

    # Show colorbar
    plt.colorbar(label='Normalized mean of the performance in the final epoch')

    # Add original values as text
    for i in range(len(annotation)):
        for j in range(len(annotation[0])):
            plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(f"app/figures/grid_search/heatmap_log_costs.png")
    plt.show()
    plt.close()
import omnisafe
import torch
import os
import argparse
import random as rand
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import wandb
from omnisafe.utils.config import get_default_kwargs_yaml
from custom_envs.hand_made_levels.hm_curriculum_env import HMCurriculumEnv

def get_configs(folder, algos, epochs, cost_limit, seed, save_freq = None, steps_per_epoch = 1000, 
                update_iters = 1, nn_size = 256, lag_multiplier_init = 0.1, lag_multiplier_lr = 0.01):
    """
    steps_per_epoch (int): the number of steps before the policy is updated
    update_iters (int): the number of update iterations per update
    """

    if save_freq == None:
        save_freq = epochs

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

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
                # 'penalty_coef': 0.01,
            },
            'logger_cfgs': {
                'log_dir': "./app/results/" + folder,
                'save_model_freq': save_freq,
                # 'use_wandb': True,
                # 'wandb_project': "test-master-thesis",
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

def train_agent(agent, episodes = 1, render_episodes = 1, make_videos = False, epochs_to_render = []):
    agent.learn()

    agent.plot(smooth=1)

    agent.evaluate(num_episodes=episodes)

    if make_videos:
        agent.render(num_episodes=render_episodes, render_mode='rgb_array', width=256, height=256, 
                     epochs_to_render=epochs_to_render)

def plot_train(folder, curr_changes, cost_limit, repetitions, include_weak=False, use_std=False):
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
    means_baseline = {"rewards": [], "costs": []}
    means_curr = {"rewards": [], "costs": []}

    plt.figure(figsize=(10, 5), dpi=80)

    # Plot baseline rewards
    for algorithm_name in baseline_rewards_mean.keys():
        if len(means_baseline['rewards']) == 0:
            means_baseline['rewards'] = baseline_rewards_mean[algorithm_name]
        plt.plot(np.arange(1, len(baseline_rewards_mean[algorithm_name]) + 1), baseline_rewards_mean[algorithm_name], label="Baseline - " + algorithm_name.split('-')[0])
        if use_std:
            plt.fill_between(x=np.arange(1, len(baseline_rewards_mean[algorithm_name]) + 1),
                        y1=baseline_rewards_mean[algorithm_name] - baseline_rewards_std[algorithm_name],
                        y2=baseline_rewards_mean[algorithm_name] + baseline_rewards_std[algorithm_name], 
                        alpha=0.2)
        else:
            # Use standard error
            plt.fill_between(x=np.arange(1, len(baseline_rewards_mean[algorithm_name]) + 1),
                        y1=np.asarray(baseline_rewards_mean[algorithm_name]) - (np.asarray(baseline_rewards_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)),
                        y2=np.asarray(baseline_rewards_mean[algorithm_name]) + (np.asarray(baseline_rewards_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)), 
                        alpha=0.2)

    # Plot curriculum rewards
    for algorithm_name in curr_rewards_mean.keys():
        if len(means_curr['rewards']) == 0:
            means_curr['rewards'] = curr_rewards_mean[algorithm_name]
        plt.plot(np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1), curr_rewards_mean[algorithm_name], label="Curriculum Strong - " + algorithm_name.split('-')[0])
        if use_std:
            plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1),
                        y1=curr_rewards_mean[algorithm_name] - curr_rewards_std[algorithm_name],
                        y2=curr_rewards_mean[algorithm_name] + curr_rewards_std[algorithm_name], 
                        alpha=0.2)
        else:
            # Use standard error
            plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1),
                        y1=np.asarray(curr_rewards_mean[algorithm_name]) - (np.asarray(curr_rewards_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)),
                        y2=np.asarray(curr_rewards_mean[algorithm_name]) + (np.asarray(curr_rewards_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)), 
                        alpha=0.2)

        if include_weak:
            plt.plot(np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1 - last_change), curr_rewards_mean[algorithm_name][last_change:], label="Curriculum Weak - " + algorithm_name.split('-')[0])
            if use_std:
                plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1),
                            y1=curr_rewards_mean[algorithm_name] - curr_rewards_std[algorithm_name],
                            y2=curr_rewards_mean[algorithm_name] + curr_rewards_std[algorithm_name], 
                            alpha=0.2)
            else:
                # Use standard error
                plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1),
                            y1=(np.asarray(curr_rewards_mean[algorithm_name]) - (np.asarray(curr_rewards_std[algorithm_name]) / 
                                                                        (repetitions ** 0.5)))[last_change:],
                            y2=(np.asarray(curr_rewards_mean[algorithm_name]) + (np.asarray(curr_rewards_std[algorithm_name]) / 
                                                                        (repetitions ** 0.5)))[last_change:], 
                            alpha=0.2)
            # plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1 - last_change),
            #                 y1=(curr_rewards_mean[algorithm_name] - curr_rewards_std[algorithm_name])[last_change:],
            #                 y2=(curr_rewards_mean[algorithm_name] + curr_rewards_std[algorithm_name])[last_change:], alpha=0.2)

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
        if len(means_baseline['costs']) == 0:
            means_baseline['costs'] = baseline_costs_mean[algorithm_name]
        plt.plot(np.arange(1, len(baseline_costs_mean[algorithm_name]) + 1), baseline_costs_mean[algorithm_name], label="Baseline - " + algorithm_name.split('-')[0])
        if use_std:
            plt.fill_between(x=np.arange(1, len(baseline_costs_mean[algorithm_name]) + 1),
                        y1=baseline_costs_mean[algorithm_name] - baseline_costs_std[algorithm_name],
                        y2=baseline_costs_mean[algorithm_name] + baseline_costs_std[algorithm_name], 
                        alpha=0.2)
        else:
            # Use standard error
            plt.fill_between(x=np.arange(1, len(baseline_costs_mean[algorithm_name]) + 1),
                        y1=np.asarray(baseline_costs_mean[algorithm_name]) - (np.asarray(baseline_costs_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)),
                        y2=np.asarray(baseline_costs_mean[algorithm_name]) + (np.asarray(baseline_costs_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)), 
                        alpha=0.2)

    # Plot curriculum costs
    for algorithm_name in curr_costs_mean.keys():
        if len(means_curr['costs']) == 0:
            means_curr['costs'] = curr_costs_mean[algorithm_name]
        plt.plot(np.arange(1, len(curr_costs_mean[algorithm_name]) + 1), curr_costs_mean[algorithm_name], label="Curriculum Strong - " + algorithm_name.split('-')[0])
        if use_std:
            plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1),
                        y1=curr_costs_mean[algorithm_name] - curr_costs_std[algorithm_name],
                        y2=curr_costs_mean[algorithm_name] + curr_costs_std[algorithm_name], 
                        alpha=0.2)
        else:
            # Use standard error
            plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1),
                        y1=np.asarray(curr_costs_mean[algorithm_name]) - (np.asarray(curr_costs_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)),
                        y2=np.asarray(curr_costs_mean[algorithm_name]) + (np.asarray(curr_costs_std[algorithm_name]) / 
                                                                    (repetitions ** 0.5)), 
                        alpha=0.2)

        if include_weak:
            plt.plot(np.arange(1, len(curr_costs_mean[algorithm_name]) + 1 - last_change), curr_costs_mean[algorithm_name][last_change:], label="Curriculum Weak - " + algorithm_name.split('-')[0])
            if use_std:
                plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1),
                            y1=curr_costs_mean[algorithm_name] - curr_costs_std[algorithm_name],
                            y2=curr_costs_mean[algorithm_name] + curr_costs_std[algorithm_name], 
                            alpha=0.2)
            else:
                # Use standard error
                plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1),
                            y1=(np.asarray(curr_costs_mean[algorithm_name]) - (np.asarray(curr_costs_std[algorithm_name]) / 
                                                                        (repetitions ** 0.5)))[last_change:],
                            y2=(np.asarray(curr_costs_mean[algorithm_name]) + (np.asarray(curr_costs_std[algorithm_name]) / 
                                                                        (repetitions ** 0.5)))[last_change:], 
                            alpha=0.2)
            # plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1 - last_change),
            #                 y1=(curr_costs_mean[algorithm_name] - curr_costs_std[algorithm_name])[last_change:],
            #                 y2=(curr_costs_mean[algorithm_name] + curr_costs_std[algorithm_name])[last_change:], alpha=0.2)

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

    return means_baseline, means_curr

def plot_eval(folder, curr_changes, cost_limit, repetitions, eval_episodes, use_std = False):
    def extract_values(pattern, text):
        return [float(match.group(1)) for match in re.finditer(pattern, text)]

    def process_data(algorithms, directory, rewards_mean, costs_mean, lengths_mean, successes_mean, 
                     rewards_std, costs_std, lengths_std, successes_std, indices):
        for algorithm in algorithms:
            seed_paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            eval_paths = [os.path.join(path, "evaluation") for path in seed_paths]

            epoch_data = {}

            for path in eval_paths:
                epochs = [entry.name for entry in os.scandir(path)]

                for epoch in epochs:
                    with open(os.path.join(path, epoch, "result.txt"), 'r') as file:
                        data = file.read()

                        rewards = extract_values(r'Episode reward: ([\d\.-]+)', data)
                        costs = extract_values(r'Episode cost: ([\d\.-]+)', data)
                        lengths = extract_values(r'Episode length: ([\d\.-]+)', data)

                        index = int(epoch.split("-")[1])
                        if index not in epoch_data:
                            epoch_data[index] = {'rewards': [], 'costs': [], 'lengths': [], 'successes': []}
                        epoch_successes = [1 if length < 1000 and cost <= cost_limit else 0 for length, cost in zip(lengths, costs)]
                        epoch_data[index]['rewards'].append(rewards)
                        epoch_data[index]['costs'].append(costs)
                        epoch_data[index]['lengths'].append(lengths)
                        epoch_data[index]['successes'].append(epoch_successes)

            for epoch_number, data in epoch_data.items():
                rewards_mean[algorithm] = rewards_mean.get(algorithm, []) + [np.mean(data['rewards'])]
                costs_mean[algorithm] = costs_mean.get(algorithm, []) + [np.mean(data['costs'])]
                lengths_mean[algorithm] = lengths_mean.get(algorithm, []) + [np.mean(data['lengths'])]
                successes_mean[algorithm] = successes_mean.get(algorithm, []) + [np.mean(data['successes'])]
                rewards_std[algorithm] = rewards_std.get(algorithm, []) + [np.std(data['rewards'])]
                costs_std[algorithm] = costs_std.get(algorithm, []) + [np.std(data['costs'])]
                lengths_std[algorithm] = lengths_std.get(algorithm, []) + [np.std(data['lengths'])]
                successes_std[algorithm] = successes_std.get(algorithm, []) + [np.std(data['successes'])]

            if len(indices) == 0:
                for key in epoch_data.keys():
                    indices.setdefault(key)

    baseline_dir = "app/results/" + folder + "/baseline"
    curr_dir = "app/results/" + folder + "/curriculum"

    baseline_algorithms = [entry.name for entry in os.scandir(baseline_dir)]
    curr_algorithms = [entry.name for entry in os.scandir(curr_dir)]

    baseline_rewards_mean, baseline_costs_mean, baseline_lengths_mean, baseline_successes_mean = {}, {}, {}, {}
    baseline_rewards_std, baseline_costs_std, baseline_lengths_std, baseline_successes_std = {}, {}, {}, {}
    curr_rewards_mean, curr_costs_mean, curr_lengths_mean, curr_successes_mean = {}, {}, {}, {}
    curr_rewards_std, curr_costs_std, curr_lengths_std, curr_successes_std = {}, {}, {}, {}
    indices = {}

    process_data(baseline_algorithms, baseline_dir, baseline_rewards_mean, baseline_costs_mean, baseline_lengths_mean, 
                 baseline_successes_mean, baseline_rewards_std, baseline_costs_std, baseline_lengths_std,
                 baseline_successes_std, indices)
    process_data(curr_algorithms, curr_dir, curr_rewards_mean, curr_costs_mean, curr_lengths_mean, curr_successes_mean, 
                 curr_rewards_std, curr_costs_std, curr_lengths_std, curr_successes_std, indices)
    
    indices = indices.keys()

    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)

    means_baseline = {"rewards": [], "costs": [], "lengths": [], "successes": []}
    means_curr = {"rewards": [], "costs": [], "lengths": [], "successes": []}

    for data_type in ['rewards', 'costs', 'lengths', 'successes']:
        plt.figure(figsize=(10, 5), dpi=80)

        for algorithm_name in eval(f"baseline_{data_type}_mean").keys():
            sorted_values = sorted(zip(indices, eval(f"baseline_{data_type}_mean")[algorithm_name], eval(f"baseline_{data_type}_std")[algorithm_name]))
            sorted_indices, sorted_means, sorted_stds = zip(*sorted_values)
            if len(means_baseline[data_type]) == 0:
                means_baseline[data_type] = sorted_means
            plt.plot(sorted_indices, sorted_means, label=f"Baseline - {algorithm_name.split('-')[0]}")
            if use_std:
                plt.fill_between(x=sorted_indices,
                            y1=np.asarray(sorted_means) - np.asarray(sorted_stds),
                            y2=np.asarray(sorted_means) + np.asarray(sorted_stds), 
                            alpha=0.2)
            else:
                # Use standard error
                plt.fill_between(x=sorted_indices,
                            y1=np.asarray(sorted_means) - (np.asarray(sorted_stds) / ((repetitions * eval_episodes) ** 0.5)),
                            y2=np.asarray(sorted_means) + (np.asarray(sorted_stds) / ((repetitions * eval_episodes) ** 0.5)), 
                            alpha=0.2)

        for algorithm_name in eval(f"curr_{data_type}_mean").keys():
            sorted_values = sorted(zip(indices, eval(f"curr_{data_type}_mean")[algorithm_name], eval(f"curr_{data_type}_std")[algorithm_name]))
            sorted_indices, sorted_means, sorted_stds = zip(*sorted_values)
            if len(means_curr[data_type]) == 0:
                means_curr[data_type] = sorted_means
            plt.plot(sorted_indices, sorted_means, label=f"Curriculum Strong - {algorithm_name.split('-')[0]}")
            if use_std:
                plt.fill_between(x=sorted_indices,
                            y1=np.asarray(sorted_means) - np.asarray(sorted_stds),
                            y2=np.asarray(sorted_means) + np.asarray(sorted_stds), 
                            alpha=0.2)
            else:
                # Use standard error
                plt.fill_between(x=sorted_indices,
                            y1=np.asarray(sorted_means) - (np.asarray(sorted_stds) / ((repetitions * eval_episodes) ** 0.5)),
                            y2=np.asarray(sorted_means) + (np.asarray(sorted_stds) / ((repetitions * eval_episodes) ** 0.5)), 
                            alpha=0.2)

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

    return means_baseline, means_curr

def print_eval(folder, means_baseline, means_curr, eval_means_baseline, eval_means_curr, save_freq):
    for means, eval_means, agent_type in zip([means_baseline, means_curr], 
                                             [eval_means_baseline, eval_means_curr],
                                             ["baseline", "curriculum"]):
        reward = means["rewards"][-1]
        cost = means["costs"][-1]
        eval_reward = eval_means["rewards"][-1]
        eval_cost = eval_means["costs"][-1]
        eval_length = eval_means["lengths"][-1]
        eval_success = eval_means["successes"][-1]
        auc_cost = np.trapz(means["costs"], dx=1)
        auc_eval_cost = np.trapz(eval_means["costs"], dx=save_freq)

        with open(os.path.join(f"app/figures/{folder}/", f"{agent_type}-metrics.txt"), 'w') as file:
            file.write("Last epoch results:\n")
            file.write(f"Reward: {reward}\n")
            file.write(f"Cost: {cost}\n")
            file.write(f"Evaluation reward: {eval_reward}\n")
            file.write(f"Evaluation cost: {eval_cost}\n")
            file.write(f"Evaluation episode length: {eval_length}\n")
            file.write(f"Evaluation success rate: {eval_success}\n")
            file.write("\nAll epochs results:\n")
            file.write(f"AUC of the cost curve: {auc_cost}\n")
            file.write(f"AUC of the evaluation cost curve: {auc_eval_cost}\n")
            file.close()

def run_experiment(eval_episodes, render_episodes, cost_limit, seed, save_freq, epochs, baseline_algorithms, curr_algorithms, folder_base):
    # Get configurations
    base_cfgs = get_configs(folder=folder_base + "/baseline", algos=baseline_algorithms, epochs=epochs, 
                            cost_limit=cost_limit, seed=seed, save_freq = save_freq)
    curr_cfgs = get_configs(folder=folder_base + "/curriculum", algos=curr_algorithms, epochs=epochs, 
                            cost_limit=cost_limit, seed=seed, save_freq = save_freq)

    # Initialize agents
    baseline_env_id = 'SafetyPointHM3-v0'
    curr_env_id = 'SafetyPointHM0-v0'

    baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
    curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

    # Train agents
    for baseline_agent in baseline_agents:
        train_agent(baseline_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

    for curriculum_agent in curriculum_agents:
        train_agent(curriculum_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

if __name__ == '__main__':
    # wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")

    eval_episodes = 1#5
    render_episodes = 1#3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 1#10
    epochs = 1#800
    repetitions = 10#10
    baseline_algorithms = ["PPOLag"] # ["PPO", "PPOLag", "P3O"]
    curr_algorithms = ["PPOLag"] # ["PPOEarlyTerminated", "PPOLag", "CPPOPID", "CPO", "IPO", "P3O"]
    folder_base = "longer_training/half_curr"
    curr_changes = [10, 20, 30]
    seeds = [int(rand.random() * 10000) for i in range(repetitions)]

    def use_params(seed):
        run_experiment(eval_episodes, render_episodes, cost_limit, seed, steps_per_epoch, save_freq, epochs, baseline_algorithms, curr_algorithms, folder_base)

    # Repeat experiments
    with Pool(4) as p:
        p.map(use_params, seeds)

    # Plot the results
    means_baseline, means_curr = plot_train(folder_base, curr_changes, cost_limit, repetitions, include_weak=False)
    eval_means_baseline, eval_means_curr = plot_eval(folder_base, curr_changes, cost_limit, repetitions, eval_episodes)
    print_eval(folder_base, means_baseline, means_curr, eval_means_baseline, eval_means_curr, save_freq)
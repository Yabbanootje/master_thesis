import omnisafe
import torch
import os
import argparse
import random as rand
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing.pool import Pool
import wandb
from itertools import product
from omnisafe.utils.config import get_default_kwargs_yaml
from custom_envs.hand_made_levels.hm_curriculum_env import HMCurriculumEnv

def get_configs(folder, algos, epochs, cost_limit, seed, save_freq = None, steps_per_epoch = 1000, 
                update_iters = 1, nn_size = 256, lag_multiplier_init = 0.1, lag_multiplier_lr = 0.035,
                focops_eta = 0.02, focops_lam = 1.5):
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
                # 'penalty_coef': 0.05,
            },
            'logger_cfgs': {
                'log_dir': "./app/results/" + folder,
                'save_model_freq': save_freq,
                'use_wandb': True,
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
                print("The algo_folder found is:", algo_folder)
                algo_path = os.path.join("app/results/", folder, algo_folder)
                seed_folder = [fldr for fldr in os.listdir(algo_path) if "seed-" + str(cfg.get("seed")).zfill(3) in fldr][0]
                print("The seed_folder found is:", seed_folder)
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

def plot_train(folder, curr_changes, cost_limit, include_weak=False, include_seeds=False, use_std=False):
    # Get folder names for all algorithms
    baseline_dir = "app/results/" + folder + "/baseline"
    curr_dir = "app/results/" + folder + "/curriculum"

    # Function to read progress csv and concatenate
    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            for path in paths:
                df = pd.read_csv(os.path.join(path, "progress.csv")).rename(columns=
                    {"Metrics/EpRet": "return", "Metrics/EpCost": "cost", "Metrics/EpLen": "length"}
                )[['return', 'cost', 'length']]
                df['Algorithm'] = algorithm.split("-")[0]
                df['type'] = algorithm_type
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-1].split("-")[1]
                df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)
                df = df.sort_index()
                df['regret'] = df['regret_per_epoch'].cumsum()
                dfs.append(df)
        return pd.concat(dfs)

    baseline_df = read_and_concat(baseline_dir, os.listdir(baseline_dir), 'baseline')
    curr_df = read_and_concat(curr_dir, os.listdir(curr_dir), 'curriculum')

    # Combine both baseline and curriculum dataframes
    combined_df = pd.concat([baseline_df, curr_df]).reset_index(names="step")
    
    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)
        
    last_change = curr_changes[-1]

    for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret']:
        # Plotting using Seaborn
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5), dpi=200)

        # include a zoomed in cost curve
        zoomed = ""
        if metric == 'cost_zoom':
            metric = 'cost'
            plt.ylim(0, 2 * cost_limit)
            zoomed = "_zoom"

        if metric == 'cost':
            plt.axhline(y=cost_limit, color='black', linestyle='-')
        if metric == 'regret':
            x = range(combined_df["step"].max())
            y = [cost_limit * x_val for x_val in x]
            plt.plot(x, y, color='black', linestyle=':')
        sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
        if include_seeds:
            ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', estimator=None, legend=False)

        for change in curr_changes:
            plt.axvline(x=change, color="gray", linestyle='-')

        plt.legend(loc=(1.01, 0.01), ncol=1)
        if include_seeds:
            plt.setp(ax.lines[2:], alpha=0.2)
        plt.tight_layout(pad=2)
        plt.title(f"{metric.replace('_', ' ').capitalize()}s of agents using curriculum and baseline agent")
        plt.xlabel("x1000 Steps")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.savefig(f"app/figures/{folder}/{metric}s{zoomed}.png")
        plt.show()
        plt.close()

    return combined_df

def plot_eval(folder, curr_changes, cost_limit, save_freq, include_weak=False, include_seeds=False, include_repetitions=False, use_std=False):
    def extract_values(pattern, text):
        return [float(match.group(1)) for match in re.finditer(pattern, text)]

    baseline_dir = "app/results/" + folder + "/baseline"
    curr_dir = "app/results/" + folder + "/curriculum"

    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            seed_paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            eval_paths = [os.path.join(path, "evaluation") for path in seed_paths]

            for path in eval_paths:
                epochs = [entry.name for entry in os.scandir(path)]

                returns = []
                costs = []
                lengths = []
                steps = []

                reps = 0

                for epoch in epochs:
                    with open(os.path.join(path, epoch, "result.txt"), 'r') as file:
                        data = file.read()

                        return_ = extract_values(r'Episode reward: ([\d\.-]+)', data)
                        cost_ = extract_values(r'Episode cost: ([\d\.-]+)', data)
                        length_ = extract_values(r'Episode length: ([\d\.-]+)', data)

                        returns += return_
                        costs += cost_
                        lengths += length_

                        if reps == 0:
                            reps = len(return_)

                        index = int(epoch.split("-")[1])
                        steps += [index for i in range(reps)]

                df = pd.DataFrame({'return': returns, 'cost': costs, 'length': lengths, 'step': steps})
                df['Algorithm'] = algorithm.split("-")[0]
                df['type'] = algorithm_type
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-2].split("-")[1]
                df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)

                df['step'] = pd.to_numeric(df['step'])
                df = df.sort_values(by=['step']).reset_index()
                avg_regret_per_epoch= df.groupby(df.index // reps)['regret_per_epoch'].mean()
                df['regret'] = avg_regret_per_epoch.cumsum().repeat(reps).reset_index(drop=True)
                dfs.append(df)
        return pd.concat(dfs)

    baseline_algorithms = os.listdir(baseline_dir)
    curr_algorithms = os.listdir(curr_dir)

    baseline_df = read_and_concat(baseline_dir, baseline_algorithms, 'baseline')
    curr_df = read_and_concat(curr_dir, curr_algorithms, 'curriculum')

    combined_df = pd.concat([baseline_df, curr_df]).reset_index(drop=True)
    
    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)

    for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret']:
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5), dpi=200)
        
        # include a zoomed in cost curve
        zoomed = ""
        if metric == 'cost_zoom':
            metric = 'cost'
            plt.ylim(0, 2 * cost_limit)
            zoomed = "_zoom"

        if metric == 'cost':
            plt.axhline(y=cost_limit, color='black', linestyle='-')
        if metric == 'regret':
            x = range(0, combined_df["step"].max() + save_freq, save_freq)
            y = [cost_limit * i for i in range(len(x))]
            plt.plot(x, y, color='black', linestyle=':')
        sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
        if include_seeds:
            if include_repetitions:
                ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', errorbar=None, estimator=None, legend=False)
            else:
                ax = sns.lineplot(data=combined_df.groupby(["step", "Algorithm", "type", "seed"]).mean(), x='step', y=metric, hue='Algorithm', 
                                  style='type', units='seed', errorbar=None, estimator=None, legend=False)

        for change in curr_changes:
            plt.axvline(x=change, color="gray", linestyle='-')

        plt.legend(loc=(1.01, 0.01), ncol=1)
        if include_seeds:
            plt.setp(ax.lines[2:], alpha=0.2)
        plt.tight_layout(pad=2)
        plt.title(f"{metric.replace('_', ' ').capitalize() if metric != 'length' else 'Episode' + metric}s of agents using curriculum and baseline agent during evalutaion")
        plt.xlabel("x1000 Steps")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.savefig(f"app/figures/{folder}/{metric}s{zoomed}_eval.png")
        plt.show()
        plt.close()

    return combined_df

def print_eval(folder, train_df, eval_df, save_freq, cost_limit):
    for (algorithm, algorithm_type), filtered_train_df in train_df.groupby(["Algorithm", 'type']):
        filtered_eval_df = eval_df[(eval_df["Algorithm"] == algorithm) & (eval_df['type'] == algorithm_type)]
        mean_train_df = filtered_train_df.groupby(["step"]).mean(numeric_only=True)
        mean_eval_df = filtered_eval_df.groupby(["step"]).mean(numeric_only=True)
        return_ = mean_train_df["return"].iloc[-1]
        cost = mean_train_df["cost"].iloc[-1]
        eval_return = mean_eval_df["return"].iloc[-1]
        eval_cost = mean_eval_df["cost"].iloc[-1]
        eval_length = mean_eval_df["length"].iloc[-1]
        auc_cost = np.trapz(mean_train_df["cost"], dx=1)
        auc_eval_cost = np.trapz(mean_eval_df["cost"], dx=save_freq)
        regret_train = mean_train_df["regret"].iloc[-1]
        regret_eval = mean_eval_df["regret"].iloc[-1]

        if not os.path.isdir(f"app/figures/{folder}/{algorithm_type}_metrics"):
            os.makedirs(f"app/figures/{folder}/{algorithm_type}_metrics")
        with open(os.path.join(f"app/figures/{folder}/", f"{algorithm_type}_metrics/{algorithm}-metrics.txt"), 'w') as file:
            file.write("Last epoch results:\n")
            file.write(f"Return: {return_}\n")
            file.write(f"Cost: {cost}\n")
            file.write(f"Evaluation return: {eval_return}\n")
            file.write(f"Evaluation cost: {eval_cost}\n")
            file.write(f"Evaluation episode length: {eval_length}\n")
            file.write("\nAll epochs results:\n")
            file.write(f"AUC of the cost curve: {auc_cost}\n")
            file.write(f"AUC of the evaluation cost curve: {auc_eval_cost}\n")
            file.write(f"Cost regret: {regret_train}\n")
            file.write(f"Evaluation cost regret: {regret_eval}\n")
            file.close()

def run_experiment(eval_episodes, render_episodes, cost_limit, seed, save_freq, epochs, algorithm, env_id, folder, curr_changes):
    # Get configurations
    if "HM1" in env_id or "HM2" in env_id:
        epochs = 500
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
            env_id = f'SafetyPointFrom{end_task-1}HM{end_task if end_task < 6 else "T"}-v0'
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
    repetitions = 5
    baseline_algorithms = ["FOCOPS"]
    curr_algorithms = ["FOCOPS"]
    folder_base = "incremental_static_curriculum_biglr"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [7337, 175, 4678, 9733, 3743] # [572, 5689, 3968, 7596, 5905] # [int(rand.random() * 10000) for i in range(repetitions)]

    # Repeat experiments
    wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
    for end_task in range(2, len(curr_changes) + 1):
        with Pool(8) as p:
            args_base = list(product(baseline_algorithms, [end_task], ["baseline"], seeds))
            args_curr = list(product(curr_algorithms, [end_task], ["curriculum"], seeds))
            args = args_curr + args_base
            p.starmap(use_params, args)

    # Plot the results
    train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
    eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, save_freq=save_freq)
    print_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)
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
from multiprocessing import Pool
import wandb
from itertools import product
from omnisafe.utils.config import get_default_kwargs_yaml
from custom_envs.hand_made_levels.hm_curriculum_env import HMCurriculumEnv

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copyreg
import types
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

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
                df = pd.read_csv(os.path.join(path, "progress.csv")).rename(columns={"Metrics/EpRet": "return", "Metrics/EpCost": "cost"})[['return', 'cost']]
                df['Algorithm'] = algorithm.split("-")[0]
                df['type'] = algorithm_type
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-1].split("-")[1]
                dfs.append(df)
        return pd.concat(dfs)

    baseline_df = read_and_concat(baseline_dir, os.listdir(baseline_dir), 'baseline')
    curr_df = read_and_concat(curr_dir, os.listdir(curr_dir), 'curriculum')

    # Combine both baseline and curriculum dataframes
    combined_df = pd.concat([baseline_df, curr_df]).reset_index(names="step")

    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)
        
    last_change = curr_changes[-1]

    for metric in ['return', 'cost']:
        # Plotting using Seaborn
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5), dpi=80)
        if metric == 'cost':
            plt.axhline(y=cost_limit, color='r', linestyle='-')
        sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
        if include_seeds:
            ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', estimator=None, legend=False)

        for change in curr_changes:
            plt.axvline(x=change, color="gray", linestyle='-')

        plt.legend(loc=(1.01, 0.01), ncol=1)
        if include_seeds:
            plt.setp(ax.lines[2:], alpha=0.2)
        plt.tight_layout(pad=2)
        plt.title(f"{metric.capitalize()}s of agents using curriculum and baseline agent")
        plt.xlabel("x1000 Steps")
        plt.ylabel(metric.capitalize())
        plt.savefig("app/figures/" + folder + "/" + metric + "s.png")
        plt.show()
        plt.close()

    # TODO: double check that below is expected to be returned
    return combined_df

def plot_eval(folder, curr_changes, cost_limit, include_weak=False, include_seeds=False, include_repetitions=False, use_std=False):
    def extract_values(pattern, text):
        return [float(match.group(1)) for match in re.finditer(pattern, text)]

    baseline_dir = "app/results/" + folder + "/baseline"
    curr_dir = "app/results/" + folder + "/curriculum"

    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            seed_paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            eval_paths = [os.path.join(path, "evaluation") for path in seed_paths]
            print(eval_paths)

            for path in eval_paths:
                epochs = [entry.name for entry in os.scandir(path)]

                returns = []
                costs = []
                lengths = []
                steps = []

                for epoch in epochs:
                    with open(os.path.join(path, epoch, "result.txt"), 'r') as file:
                        data = file.read()

                        return_ = extract_values(r'Episode reward: ([\d\.-]+)', data)
                        cost_ = extract_values(r'Episode cost: ([\d\.-]+)', data)
                        length_ = extract_values(r'Episode length: ([\d\.-]+)', data)

                        returns += return_
                        costs += cost_
                        lengths += length_

                        index = int(epoch.split("-")[1])
                        steps += [index for i in range(len(return_))]

                df = pd.DataFrame({'return': returns, 'cost': costs, 'length': lengths, 'step': steps})
                df['Algorithm'] = algorithm.split("-")[0]
                df['type'] = algorithm_type
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-2].split("-")[1]
                dfs.append(df)
        return pd.concat(dfs)

    baseline_algorithms = os.listdir(baseline_dir)
    curr_algorithms = os.listdir(curr_dir)

    baseline_df = read_and_concat(baseline_dir, baseline_algorithms, 'baseline')
    curr_df = read_and_concat(curr_dir, curr_algorithms, 'curriculum')

    combined_df = pd.concat([baseline_df, curr_df]).reset_index(drop=True)

    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)

    for metric in ['return', 'cost', 'length']:
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5), dpi=80)
        if metric == 'cost':
            plt.axhline(y=cost_limit, color='r', linestyle='-')
        sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
        if include_seeds:
            if include_repetitions:
                ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', errorbar=None, estimator=None, legend=False)
            else:
                print(combined_df.groupby(["step", "Algorithm", "type", "seed"]).mean())
                ax = sns.lineplot(data=combined_df.groupby(["step", "Algorithm", "type", "seed"]).mean(), x='step', y=metric, hue='Algorithm', 
                                  style='type', units='seed', errorbar=None, estimator=None, legend=False)

        for change in curr_changes:
            plt.axvline(x=change, color="gray", linestyle='-')

        plt.legend(loc=(1.01, 0.01), ncol=1)
        if include_seeds:
            plt.setp(ax.lines[2:], alpha=0.2)
        plt.tight_layout(pad=2)
        plt.title(f"{metric.capitalize() if metric != 'length' else 'Episode' + metric}s of agents using curriculum and baseline agent during evalutaion")
        plt.xlabel("x1000 Steps")
        plt.ylabel(metric.capitalize())
        plt.savefig("app/figures/" + folder + "/" + metric + "s_eval.png")
        plt.show()
        plt.close()

    return combined_df

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

def run_experiment(eval_episodes, render_episodes, cost_limit, seed, save_freq, epochs, algorithm, env_id, folder):
    # Get configurations
    cfgs = get_configs(folder=folder, algos=[algorithm], epochs=epochs, cost_limit=cost_limit, seed=seed, 
                       save_freq = save_freq)

    # Initialize agents
    agents = get_agents([algorithm], env_id, cfgs)

    # Train agents
    for agent in agents:
        train_agent(agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

class Test():
    def use_params(algorithm, type, seed):
        if type == "baseline":
            env_id = 'SafetyPointHM4-v0'
        elif type == "curriculum":
            env_id = 'SafetyPointHM0-v0'
        else:
            raise Exception("Invalid type, must be either 'baseline' or 'curriculum'.")

        run_experiment(eval_episodes=eval_episodes, render_episodes=render_episodes, cost_limit=cost_limit, 
                        seed=seed, save_freq=save_freq, epochs=epochs, algorithm=algorithm, 
                        env_id=env_id, folder=folder_base + "/" + type)
        
def use_params(test, algorithm, type, seed):
    test.use_params(algorithm, type, seed)

if __name__ == '__main__':
    wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")

    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 800
    repetitions = 5
    baseline_algorithms = ["PPO", "PPOLag", "CPO"]
    curr_algorithms = ["OnCRPO", "CUP", "FOCOPS", "PCPO", "PPOEarlyTerminated"]
    folder_base = "algorithm_comparison"
    curr_changes = [10, 20, 40, 100]
    seeds = [int(rand.random() * 10000) for i in range(repetitions)]

    test = Test()

    # Repeat experiments
    with Pool(8) as p:
        args_base = list(product(baseline_algorithms, ["baseline"], seeds))
        args_curr = list(product(curr_algorithms, ["curriculum"], seeds))
        args = args_curr + args_base
        p.starmap(test.use_params, args)

    # Plot the results
    train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
    eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    # print_eval(folder_base, means_baseline, means_curr, eval_means_baseline, eval_means_curr, save_freq)
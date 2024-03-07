import omnisafe
import torch
import os
import random as rand
import pandas as pd
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
    safe_freq = epochs

    custom_cfgs = []

    for algo in algos:
        # cfg_path = get_yaml_path(algo, "on-policy")
        # kwargs = load_yaml(cfg_path)
        kwargs = get_default_kwargs_yaml(algo, None, "on-policy").todict()

        if random:
            seed = int(rand.random() * 10000)
        else:
            seed = 0

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
        }

        # Add cost_limit depending on specific algorithm
        if kwargs.get("lagrange_cfgs"):
            custom_cfg.update({'lagrange_cfgs': {
                'cost_limit': cost_limit,
            },
            })
        if kwargs["algo_cfgs"].get("cost_limit"):
            custom_cfg["algo_cfgs"].update({'cost_limit': cost_limit,})

        custom_cfgs.append(custom_cfg)

    return custom_cfgs

def get_agents(algorithms, env_id, cfgs):
    agents = []
    for algorithm, cfg in zip(algorithms, cfgs):
        agents.append(omnisafe.Agent(algorithm, env_id, custom_cfgs=cfg))

    return agents

def train_agent(agent, videos = 1):
    agent.learn()

    agent.plot(smooth=1)

    agent.evaluate(num_episodes=videos)

    agent.render(num_episodes=videos, render_mode='rgb_array', width=256, height=256)

def nice_plot(folder, curr_changes, cost_limit, include_weak=False):
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

    # script_dir = os.path.dirname(__file__)
    # results_dir = os.path.join(script_dir, 'Results/')

    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)

    # std_rewards = np.std(rewards, axis=0)
    # std_costs = np.std(costs, axis=0)
    # std_rewardsc = np.std(rewardsc, axis=0)
    # std_costsc = np.std(costsc, axis=0)

    # ticks = np.arange(len(mean_rewards), step=5)
    # ticks[0] = 1

    plt.figure()

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
            plt.plot(np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1 - 20), curr_rewards_mean[algorithm_name][20:], label="Curriculum Weak - " + algorithm_name.split('-')[0])
            plt.fill_between(x=np.arange(1, len(curr_rewards_mean[algorithm_name]) + 1 - 20),
                            y1=(curr_rewards_mean[algorithm_name] - curr_rewards_std[algorithm_name])[20:],
                            y2=(curr_rewards_mean[algorithm_name] + curr_rewards_std[algorithm_name])[20:], alpha=0.2)

    for change in curr_changes:
        plt.axvline(x=change, color="gray", linestyle='-')

    plt.legend()
    plt.grid()
    plt.title("Rewards of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Reward")
    plt.savefig("app/figures/" + folder + "/rewards_new.png")
    plt.show()

    plt.figure()

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
            plt.plot(np.arange(1, len(curr_costs_mean[algorithm_name]) + 1 - 30), curr_costs_mean[algorithm_name][30:], label="Curriculum Weak - " + algorithm_name.split('-')[0])
            plt.fill_between(x=np.arange(1, len(curr_costs_mean[algorithm_name]) + 1 - 30),
                            y1=(curr_costs_mean[algorithm_name] - curr_costs_std[algorithm_name])[30:],
                            y2=(curr_costs_mean[algorithm_name] + curr_costs_std[algorithm_name])[30:], alpha=0.2)

    plt.axhline(y=cost_limit, color='r', linestyle='-')

    for change in curr_changes:
        plt.axvline(x=change, color="gray", linestyle='-')

    plt.legend()
    plt.grid()
    plt.title("Costs of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Cost")
    plt.savefig("app/figures/" + folder + "/costs_new.png")
    plt.show()

if __name__ == '__main__':
    num_videos = 3
    cost_limit = 10.0
    epochs = 50
    repetitions = 3
    baseline_algorithms = ["PPOLag"]
    curr_algorithms = ["PPOEarlyTerminated", "PPOLag", "CPPOPID", "CPO", "IPO", "P3O"]

    # Create folder
    folder = "test-half_curriculum-multi_algos"
    folder_name = folder + "---" + str(datetime.datetime.now()).replace(' ', '-')

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
            train_agent(baseline_agent, num_videos)

        for curriculum_agent in curriculum_agents:
            train_agent(curriculum_agent, num_videos)

    # Plot the results
    curr_changes = [10, 20, 30]
    nice_plot(folder_name, curr_changes, cost_limit, include_weak=False)
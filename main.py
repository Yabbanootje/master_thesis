import omnisafe
import torch
import os
import random as rand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from safety_gymnasium.utils.registration import register
from custom_envs.curriculum_env import CurriculumEnv

steps_per_epoch = 1000
epochs = 1

def test(random, folder, num_videos):
    baseline_env_id = 'SafetyPointCurriculum2-v0'
    curr_env_id = 'SafetyPointCurriculum2-v0'

    if random:
        seed = int(rand.random() * 1000)
    else:
        seed = 0

    baseline_custom_cfgs = {
        'seed': seed,
        'train_cfgs': {
            'total_steps': epochs * steps_per_epoch,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': steps_per_epoch,
            'update_iters': 1,
            # 'cost_limit': 10.0,
            'penalty_coef': 0.01,
        },
        'logger_cfgs': {
            'log_dir': "./app/results/" + folder + "/baseline",
            'save_model_freq': epochs,
            # 'use_wandb': True,
            # 'wandb_project': "TODO",
        },
        'lagrange_cfgs': {
            'cost_limit': 10.0,
        },
    }

    curr_custom_cfgs = {
        'seed': seed,
        'train_cfgs': {
            'total_steps': epochs * steps_per_epoch,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': steps_per_epoch,
            'update_iters': 1,
            # 'cost_limit': 10.0,
            'penalty_coef': 0.01, # use costs also as a negative reward
            # 'use_cost': True, # mainly for updating the cost-critic
        },
        'logger_cfgs': {
            'log_dir': "./app/results/" + folder + "/curriculum",
            'save_model_freq': epochs,
            # 'use_wandb': True,
            # 'wandb_project': "TODO",
        },
        'lagrange_cfgs': {
            'cost_limit': 10.0,
        },
    }

    baseline_agent = omnisafe.Agent('PPOLag', baseline_env_id, custom_cfgs=baseline_custom_cfgs)
    curr_agent = omnisafe.Agent('PPOLag', curr_env_id, custom_cfgs=curr_custom_cfgs)

    # def print_agent_params(agent):
    #     scan_dir = os.scandir(os.path.join(agent.agent.logger.log_dir, 'torch_save'))
    #     for item in scan_dir:
    #         if item.is_file() and item.name.split('.')[-1] == 'pt':
    #             model_path = os.path.join(agent.agent.logger.log_dir, 'torch_save', item.name)
    #             model_params = torch.load(model_path)
    #             for thingy in model_params['pi']:
    #                 print(thingy, model_params['pi'][thingy].size())

    # print_agent_params(agent=agent)

    def get_multiple_videos(agent, episodes = 1):
        agent.learn()

        agent.plot(smooth=1)

        agent.evaluate(num_episodes=episodes)

        agent.render(num_episodes=episodes, render_mode='rgb_array', width=256, height=256)

    get_multiple_videos(agent=baseline_agent, episodes=num_videos)
    get_multiple_videos(agent=curr_agent, episodes=num_videos + 1)

def nice_plot(folder, include_weak=False):
    baseline_algorithms = os.scandir("app/results/" + folder + "/baseline")
    curr_algorithms = os.scandir("app/results/" + folder + "/curriculum")

    baseline_algorithm = str([algorithm.name for algorithm in baseline_algorithms][0])
    curr_algorithm = str([algorithm.name for algorithm in curr_algorithms][0])

    baseline_paths = os.scandir("app/results/" + folder + "/baseline/" + baseline_algorithm)
    curr_paths = os.scandir("app/results/" + folder + "/curriculum/" + curr_algorithm)

    dfs = []
    for path in baseline_paths:
        dfs.append(pd.read_csv("app/results/" + folder + "/baseline/" + baseline_algorithm + "/" + str(path.name) + "/progress.csv"))

    dfcs = []
    for path in curr_paths:
        dfcs.append(pd.read_csv("app/results/" + folder + "/curriculum/" + curr_algorithm + "/" + str(path.name) + "/progress.csv"))

    rewards = []
    costs = []

    for df in dfs:
        rewards.append(df['Metrics/EpRet'].to_numpy())
        costs.append(df['Metrics/EpCost'].to_numpy())

    rewardsc = []
    costsc = []

    for dfc in dfcs:
        rewardsc.append(dfc['Metrics/EpRet'].to_numpy())
        costsc.append(dfc['Metrics/EpCost'].to_numpy())

    # script_dir = os.path.dirname(__file__)
    # results_dir = os.path.join(script_dir, 'Results/')

    if not os.path.isdir("app/figures/" + folder):
        os.makedirs("app/figures/" + folder)

    mean_rewards = np.mean(rewards, axis=0)
    mean_costs = np.mean(costs, axis=0)
    mean_rewardsc = np.mean(rewardsc, axis=0)
    mean_costsc = np.mean(costsc, axis=0)

    std_rewards = np.std(rewards, axis=0)
    std_costs = np.std(costs, axis=0)
    std_rewardsc = np.std(rewardsc, axis=0)
    std_costsc = np.std(costsc, axis=0)

    # ticks = np.arange(len(mean_rewards), step=5)
    # ticks[0] = 1

    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_rewards, label = "Baseline")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_rewards-std_rewards, y2=mean_rewards+std_rewards, alpha=0.2)
    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_rewardsc, label = "Curriculum Strong")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_rewardsc-std_rewardsc, y2=mean_rewardsc+std_rewardsc, alpha=0.2)
    if include_weak:
        plt.plot(np.arange(1, len(mean_rewards) + 1 - 20), mean_rewardsc[20:], label = "Curriculum Weak")
        plt.fill_between(x=np.arange(1, len(mean_rewards) + 1 - 20), y1=(mean_rewardsc-std_rewardsc)[20:], y2=(mean_rewardsc+std_rewardsc)[20:], alpha=0.2)
    plt.axvline(x = 10, color = "gray", linestyle = '-')
    plt.axvline(x = 20, color = "gray", linestyle = '-')
    plt.legend()
    plt.grid()
    plt.title("Rewards of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Reward")
    plt.savefig("app/figures/" + folder + "/rewards.png")
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_costs, label = "Baseline")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_costs-std_costs, y2=mean_costs+std_costs, alpha=0.2)
    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_costsc, label = "Curriculum Strong")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_costsc-std_costsc, y2=mean_costsc+std_costsc, alpha=0.2)
    if include_weak:
        plt.plot(np.arange(1, len(mean_rewards) + 1 - 20), mean_costsc[20:], label = "Curriculum Weak")
        plt.fill_between(x=np.arange(1, len(mean_rewards) + 1 - 20), y1=(mean_costsc-std_costsc)[20:], y2=(mean_costsc+std_costsc)[20:], alpha=0.2)
    plt.axhline(y = 10, color = 'r', linestyle = '-') 
    plt.axvline(x = 10, color = "gray", linestyle = '-')
    plt.axvline(x = 20, color = "gray", linestyle = '-')
    plt.legend()
    plt.grid()
    plt.title("Costs of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Cost")
    plt.savefig("app/figures/" + folder + "/costs.png")
    plt.show()

if __name__ == '__main__':
    folder = "test-fork_server"
    num_videos = 1
    for i in range(1):
        test(True, folder, num_videos)
    # nice_plot(folder, include_weak=True)
import omnisafe
import torch
import os
import safety_gymnasium
import random as rand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from safety_gymnasium.utils.registration import register
from custom_envs.curriculum_env import CurriculumEnv

steps_per_epoch = 1000
epochs = 40

def test(random, folder):
    # register(id="Curriculum0-v0", entry_point="custom_envs.curriculum_level_0:CurriculumLevel0")

    # safety_gymnasium.vector.make(env_id="Curriculum0-v0", num_envs=1)

    baseline_env_id = 'SafetyPointBaseline2-v0'
    curr_env_id = 'SafetyPointCurriculum2-v0'

    if random:
        seed = int(rand.random() * 1000)
    else:
        seed = 0

    # env_id = "SafetyPointGoal0-v0"
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
            'cost_limit': 10.0,
        },
        'logger_cfgs': {
            'log_dir': "./app/results/" + folder + "/baseline",
            'save_model_freq': epochs,
            # 'use_wandb': True,
            # 'wandb_project': "TODO",
        },
        # 'lagrange_cfgs': {
        #     'cost_limit': 10.0,
        # },
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
            'cost_limit': 10.0,
        },
        'logger_cfgs': {
            'log_dir': "./app/results/" + folder + "/curriculum",
            'save_model_freq': epochs,
            # 'use_wandb': True,
            # 'wandb_project': "TODO",
        },
        # 'lagrange_cfgs': {
        #     'cost_limit': 10.0,
        # },
    }

    baseline_agent = omnisafe.Agent('PPOEarlyTerminated', baseline_env_id, custom_cfgs=baseline_custom_cfgs)
    curr_agent = omnisafe.Agent('PPOEarlyTerminated', curr_env_id, custom_cfgs=curr_custom_cfgs)

    def print_agent_params(agent):
        scan_dir = os.scandir(os.path.join(agent.agent.logger.log_dir, 'torch_save'))
        for item in scan_dir:
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                model_path = os.path.join(agent.agent.logger.log_dir, 'torch_save', item.name)
                model_params = torch.load(model_path)
                for thingy in model_params['pi']:
                    print(thingy, model_params['pi'][thingy].size())

    # print_agent_params(agent=agent)

    def get_multiple_videos(agent, episodes = 1):
        agent.learn()

        agent.plot(smooth=1)

        agent.evaluate(num_episodes=episodes)

        agent.render(num_episodes=episodes, render_mode='rgb_array', width=256, height=256)

    get_multiple_videos(agent=baseline_agent, episodes=3)
    get_multiple_videos(agent=curr_agent, episodes=4)

    # How is the starting position of the agent determined?

def nice_plot(folder):
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

    mean_rewards = np.mean(rewards, axis=0)
    mean_costs = np.mean(costs, axis=0)
    mean_rewardsc = np.mean(rewardsc, axis=0)
    mean_costsc = np.mean(costsc, axis=0)

    std_rewards = np.std(rewards, axis=0)
    std_costs = np.std(costs, axis=0)
    std_rewardsc = np.std(rewardsc, axis=0)
    std_costsc = np.std(costsc, axis=0)

    ticks = np.arange(len(mean_rewards), step=5)
    ticks[0] = 1

    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_rewards, label = "Baseline")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_rewards-std_rewards, y2=mean_rewards+std_rewards, alpha=0.2)
    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_rewardsc, label = "Curriculum")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_rewardsc-std_rewardsc, y2=mean_rewardsc+std_rewardsc, alpha=0.2)
    plt.legend()
    plt.title("Rewards of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Reward")
    plt.savefig("app/figures/rewards.png")
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_costs, label = "Baseline")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_costs-std_costs, y2=mean_costs+std_costs, alpha=0.2)
    plt.plot(np.arange(1, len(mean_rewards) + 1), mean_costsc, label = "Curriculum")
    plt.fill_between(x=np.arange(1, len(mean_rewards) + 1), y1=mean_costsc-std_costsc, y2=mean_costsc+std_costsc, alpha=0.2)
    plt.legend()
    plt.title("Costs of agent using curriculum and baseline agent")
    plt.xlabel("x1000 Steps")
    plt.ylabel("Cost")
    plt.savefig("app/figures/costs.png")
    plt.show()

if __name__ == '__main__':
    folder = "test-early"
    for i in range(5, folder):
        test(True)
    nice_plot(folder)
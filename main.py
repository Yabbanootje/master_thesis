import omnisafe
import torch
import os
import safety_gymnasium
# from safety_gymnasium.utils.registration import register
from custom_envs.curriculum_env import CurriculumEnv

steps_per_epoch = 1000
epochs = 3

def test():
    # register(id="Curriculum0-v0", entry_point="custom_envs.curriculum_level_0:CurriculumLevel0")

    # safety_gymnasium.vector.make(env_id="Curriculum0-v0", num_envs=1)

    env_id = 'SafetyPointCurriculum0-v0'
    env_id_1 = 'SafetyPointCurriculum1-v0'
    env_id_2 = 'SafetyPointCurriculum2-v0'
    # env_id = "SafetyPointGoal0-v0"
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': epochs * steps_per_epoch,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': steps_per_epoch,
            'update_iters': 1,
        },
        'logger_cfgs': {
            'log_dir': "./app/results",
            # 'use_wandb': True,
            # 'wandb_project': "TODO",
        },
    }

    print(env_id)

    agent = omnisafe.Agent('TRPO', env_id, custom_cfgs=custom_cfgs)
    agent_1 = omnisafe.Agent('TRPO', env_id_1, custom_cfgs=custom_cfgs)
    agent_2 = omnisafe.Agent('TRPO', env_id_2, custom_cfgs=custom_cfgs)


    def print_agent_params(agent):
        scan_dir = os.scandir(os.path.join(agent.agent.logger.log_dir, 'torch_save'))
        for item in scan_dir:
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                model_path = os.path.join(agent.agent.logger.log_dir, 'torch_save', item.name)
                model_params = torch.load(model_path)
                for thingy in model_params['pi']:
                    print(thingy, model_params['pi'][thingy].size())

    print_agent_params(agent=agent)
    print_agent_params(agent=agent_1)
    print_agent_params(agent=agent_2)

    def get_multiple_videos(agent):
        agent.learn()

        agent.plot(smooth=1)

        agent.evaluate(num_episodes=3)

        agent.render(num_episodes=3, render_mode='rgb_array', width=256, height=256)

    get_multiple_videos(agent=agent)
    get_multiple_videos(agent=agent_1)
    get_multiple_videos(agent=agent_2)

    """
    Ask a question on the github

    - Make the environment static after each episode
    - Make my own environment
        -- Do it all at once with only an environment
    - Transfer knowledge to next agent/task
    """

    # How does it calculate the rewards as of now? Because cur_level_0 does not have a step func

    # How is the starting position of the agent determined?

if __name__ == '__main__':
    test()
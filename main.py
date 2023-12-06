import omnisafe

steps_per_epoch = 1024
epochs = 3

def test():
    env_id = 'SafetyPointGoal1-v0'
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
            'use_wandb': False,
        },
    }

    agent = omnisafe.Agent('TRPO', env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=1)

    agent.evaluate(num_episodes=1)

    agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)

if __name__ == '__main__':
    test()
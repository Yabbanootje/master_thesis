import omnisafe


env_id = 'SafetyPointGoal1-v0'
custom_cfgs = {
    'train_cfgs': {
        'total_steps': 2048,
        'vector_env_nums': 1,
        'parallel': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 1024,
        'update_iters': 1,
    },
    'logger_cfgs': {
        # 'log_dir': ".\\runs",
        'use_wandb': False,
    },
}

agent = omnisafe.Agent('TRPO', env_id, custom_cfgs=custom_cfgs)
agent.learn()

agent.plot(smooth=1)
# Issue is in the fact that the file path uses \\ (instead of /) and 381 of plotter they seperate on /
# to get the filename

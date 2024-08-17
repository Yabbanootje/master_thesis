# import omnisafe
from itertools import product

baseline_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
adapt_curr_algorithms = ["PPOLag"]
folder_base = "tune_beta_kappa_reset"
curr_changes = [10, 20, 40, 100, 300, 700]
seeds = [5905, 7337, 572, 5689, 3968, 175, 4678, 9733, 3743, 7596] # [int(rand.random() * 10000) for i in range(repetitions)]
betas = [0.5, 1.0, 1.5]
kappas = [5, 10, 20]
args_curr = list(product(adapt_curr_algorithms, [6], ["adaptive_curriculum"], seeds, [2], betas, kappas))
print(args_curr)
args_curr.remove(("PPOLag", 6, "adaptive_curriculum", 3743, 2, 0.5, 20))
print(args_curr)

# env_id = 'SafetyPointGoal1-v0'
# custom_cfgs = {
#     'train_cfgs': {
#         'total_steps': 2048,
#         'vector_env_nums': 1,
#         'parallel': 1,
#     },
#     'algo_cfgs': {
#         'steps_per_epoch': 1024,
#         'update_iters': 1,
#     },
#     'logger_cfgs': {
#         # 'log_dir': ".\\runs",
#         'use_wandb': False,
#     },
# }

# agent = omnisafe.Agent('TRPO', env_id, custom_cfgs=custom_cfgs)
# agent.learn()

# agent.plot(smooth=1)
# # Issue is in the fact that the file path uses \\ (instead of /) and 381 of plotter they seperate on /
# # to get the filename

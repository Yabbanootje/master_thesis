from ...master_thesis.main import *

if __name__ == '__main__':
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 2000
    repetitions = 15
    baseline_algorithms = []#["PPO", "CPO", "OnCRPO", "CUP", "FOCOPS", "PCPO", "PPOEarlyTerminated", "PPOLag"]
    curr_algorithms = ["PPOLag"]#["OnCRPO", "CUP", "FOCOPS", "PCPO", "PPOEarlyTerminated", "PPOLag"]
    folder_base = "algorithm_comparison_extra" #"tune_beta_kappa"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [7337, 175, 4678, 9733, 3743, 572, 5689, 3968, 7596, 5905] # [int(rand.random() * 10000) for i in range(repetitions)]
    betas = [0.9, 1.0, 1.1]
    kappas = [5, 10, 20]

    # Repeat experiments
    wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
    for seed in seeds:
        with Pool(8) as p:
            args_base = list(product(baseline_algorithms, [6], ["baseline"], seeds, betas, kappas))
            args_curr = list(product(curr_algorithms, [6], ["curriculum"], seeds, betas, kappas))
            args = args_curr + args_base
            p.starmap(use_params, args)

    print("done with the standard seeds")

    for seed in [int(rand.random() * 10000) for i in range(repetitions)]:
        with Pool(8) as p:
            args_base = list(product(baseline_algorithms, [6], ["baseline"], seeds, betas, kappas))
            args_curr = list(product(curr_algorithms, [6], ["curriculum"], seeds, betas, kappas))
            args = args_curr + args_base
            p.starmap(use_params, args)

    # use_params(*("PPOLag", 4, "adaptive_curriculum", 1142, 1.1, 5))

    # Plot the results
    train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
    eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    print_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)
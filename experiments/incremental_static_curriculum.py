from ...master_thesis.main import *

if __name__ == '__main__':
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 2000
    repetitions = 5
    baseline_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
    curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    folder_base = "incremental_static_curriculum"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [int(rand.random() * 10000) for i in range(repetitions)]

    # Repeat experiments
    wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
    for end_task in range(1, len(curr_changes) + 1):
        with Pool(8) as p:
            args_base = list(product(baseline_algorithms, [end_task], ["baseline"], seeds))
            args_curr = list(product(curr_algorithms, [end_task], ["curriculum"], seeds))
            args = args_curr + args_base
            p.starmap(use_params, args)

    # Plot the results
    train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
    eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    print_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)
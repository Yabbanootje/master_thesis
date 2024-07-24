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

    # Create a figure
    fig = plt.figure(figsize=(12, 6), dpi=200)

    # Create a GridSpec with 4 rows and 3 columns
    gs = gridspec.GridSpec(4, 3)

    # Define the folders and files
    folder = folder_base
    additional_folder = "best_params"
    additional_file_text = ""

    # Define your dataframes
    dataframes = [train_df, eval_df]
    metrics = ["return", "regret", "current_task"]

    # Iterate through the dataframes and metrics
    for row_idx, combined_df in enumerate(dataframes):
        combined_df = combined_df[combined_df['beta_kappa'].isin(["1.5-5", "0.5-5", "0.5-20", "0.5-10", "1.0-20"])]

        # Create subplots for each metric
        for col_idx, metric in enumerate(metrics):
            if metric == "current_task" and row_idx == 0:
                # Place the "current_task" plot in the middle of the right column, spanning rows
                ax = plt.subplot(gs[1:3, 2])
            elif metric == "current_task":
                break
            else:
                # Place "return" and "regret" plots in the corresponding rows and columns
                ax = plt.subplot(gs[row_idx * 2:row_idx * 2 + 2, col_idx])

            # sns.set_style("whitegrid")
            sns.lineplot(data=combined_df, x='step', y=metric, hue='beta_kappa', errorbar="sd" if False else "se", ax=ax)

            if metric == 'cost':
                ax.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

            ax.set_xlabel("x1000 Steps")
            if row_idx == 0 and metric == "return":
                ax.set_ylabel("Training")
            elif row_idx == 1 and metric == "return":
                ax.set_ylabel("Evaluation")
            else:
                ax.set_ylabel("")
            ax.get_legend().remove()
            ax.set_title(metric.replace('_', ' ').capitalize())
            if metric == "current_task":
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, title="beta-kappa", loc=(1.01, 0.01), ncol=1)

    # plt.title(f"Performance of adaptive agents using different betas and kappas")
    fig.suptitle("Performance of adaptive agents using different betas and kappas")
    plt.tight_layout(pad=2)
    if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
        os.makedirs(f"figures/{folder}/{additional_folder}")
    plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.png")
    plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.pdf")
    plt.close()
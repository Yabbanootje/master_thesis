# This file corresponds to the experiments performed in section 6.4.1 of the thesis

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.plot_functions_adapt_tuning import *
from main import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 3000
    repetitions = 10
    baseline_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
    curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    adapt_curr_algorithms = ["PPOLag"]
    folder_base = "tune_beta_kappa_reset"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [5905, 7337, 572, 5689, 3968, 175, 4678, 9733, 3743, 7596] # [int(rand.random() * 10000) for i in range(repetitions)]
    betas = [0.5, 1.0, 1.5]
    kappas = [5, 10, 20]

    # Repeat experiments
    for end_task in range(6, len(curr_changes) + 1):
        with Pool(8) as p:
            args_curr = list(product(adapt_curr_algorithms, [end_task], ["adaptive_curriculum"], seeds, [exp], betas, kappas))
            p.starmap(use_params, args_curr)

    # Plot and save the results
    train_df = plot_adapt_tune_train(folder=folder_base, cost_limit=cost_limit)
    train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    eval_df = plot_adapt_tune_eval(folder=folder_base, cost_limit=cost_limit)
    eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")


    # Create a summarizing figure (Figure 6.12) with return, cost, regret and current task
    fig = plt.figure(figsize=(12, 6), dpi=200)

    # Create a GridSpec with 4 rows and 3 columns
    gs = gridspec.GridSpec(4, 3)

    # Define the folders and files
    folder = folder_base
    additional_folder = "best_params"
    additional_file_text = ""

    # Select five most promising parameter combinations
    dataframes = [train_df[(train_df["end_task"] == "T") & (train_df['beta_kappa'].isin(["1.5-5", "1.5-20", "0.5-20", "0.5-10", "1.0-20"]))], 
                  eval_df[(eval_df["end_task"] == "T") & (eval_df['beta_kappa'].isin(["1.5-5", "1.5-20", "0.5-20", "0.5-10", "1.0-20"]))]]
    
    # The metrics to be used
    metrics = ["return", "regret", "current_task"]

    # Iterate through the dataframes and metrics
    for row_idx, combined_df in enumerate(dataframes):
        # Create subplots for each metric
        for col_idx, metric in enumerate(metrics):
            if metric == "current_task" and row_idx == 0:
                # Place the "current_task" plot in the middle of the right column, spanning rows
                ax = plt.subplot(gs[0:4, 2])
            elif metric == "current_task":
                # Last column should only contain one figure
                break
            else:
                # Place "return" and "regret" plots in the corresponding rows and columns
                ax = plt.subplot(gs[row_idx * 2:row_idx * 2 + 2, col_idx])

            if metric == "regret" and row_idx == 1:
                # Cost works better than regret for evaluation data
                metric = "cost"
            if row_idx == 1:
                # Average the evaluation episodes
                combined_df[metric] = combined_df.groupby(["beta_kappa", "seed"])[metric].transform(lambda x: x.rolling(window=25, min_periods=1, center=True).mean())

            # Plot the results
            sns.lineplot(data=combined_df, x='step', y=metric, hue='beta_kappa', errorbar="sd" if False else "se", ax=ax)

            # Add cost limit
            if metric == 'cost':
                ax.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

            # Set the labels, titles and legend
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

    # Save the plot
    fig.suptitle("Performance of adaptive agents using different betas and kappas")
    plt.tight_layout(pad=2)
    if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
        os.makedirs(f"figures/{folder}/{additional_folder}")
    plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.png")
    plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.pdf")
    plt.close()


    # Create a figure that shows the task progression of the five best parameter combinations (Figure B.15)
    # Select five most promising parameter combinations
    dataframes = [train_df[(train_df["end_task"] == "T") & (train_df['beta_kappa'].isin(["1.5-5", "1.5-20", "0.5-20", "0.5-10", "1.0-20"]))], 
                  eval_df[(eval_df["end_task"] == "T") & (eval_df['beta_kappa'].isin(["1.5-5", "1.5-20", "0.5-20", "0.5-10", "1.0-20"]))]]

    # Create a FacetGrid
    sns.set_style("whitegrid")
    g = sns.FacetGrid(dataframes[0], col="beta_kappa", hue='beta_kappa', col_wrap=3, height=4)  # col_wrap controls how many plots per row

    # Map the lineplot onto the FacetGrid
    g.map_dataframe(sns.lineplot, x='step', y="current_task", errorbar="sd" if False else "se", estimator=None, units='seed', alpha=0.4)

    # Set labels and titles
    g.set_axis_labels("x1000 Steps", "Current Task", fontsize=14)
    g.set_titles("beta-kappa: {col_name}", size=14)

    # Force the x-axis ticks and labels to show on all plots
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    # Save the plot
    fig.suptitle("Performance of adaptive agents using different betas and kappas")
    plt.tight_layout(pad=2)
    if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
        os.makedirs(f"figures/{folder}/{additional_folder}")
    plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}current_task.png")
    plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}current_task.pdf")
    plt.close()
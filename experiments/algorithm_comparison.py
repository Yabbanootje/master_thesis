# This file corresponds to the experiments performed in section 6.3.1 of the thesis
# Note that the results from the thesis cannot be replicated exactly with this file,
# as the improvements made to the curriculum (such as the learning rate reset) were
# implemented these experiments and there is no way to opt-out of these improvements

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.plot_functions import plot_train, plot_eval, print_eval
from main import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 1000
    repetitions = 15
    baseline_algorithms = ["PPO", "PPOLag", "CPO", "FOCOPS", "OnCRPO", "CUP", "PCPO", "PPOEarlyTerminated"]
    curr_algorithms = ["OnCRPO", "CUP", "FOCOPS", "PCPO", "PPOEarlyTerminated", "PPOLag"]
    folder_base = "algorithm_comparison"
    curr_changes = [10, 20, 40, 100]
    seeds = [int(rand.random() * 10000) for i in range(repetitions)]

    # Repeat experiments
    wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
    with Pool(8) as p:
        args_base = list(product(baseline_algorithms, [4], ["baseline"], seeds))
        args_curr = list(product(curr_algorithms, [4], ["curriculum"], seeds))
        args = args_curr + args_base
        p.starmap(use_params, args)


    # Plot the results
    train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
    eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    print_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)

    # Save results
    train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")


    # Function to create the scatter plots that have baseline performance on the x-axis and
    # curriculum performance on the y-axis
    def plot_scatter(train_df, eval_df, exclude_outlier=True):
        # Create a figre for both the training and evaluation results
        for df, set in zip([train_df, eval_df], ["non_evaluation_", "evaluation_"]):
            # Create a subplot for each metric
            fig, axs = plt.subplots(1, 3, figsize=(13, 4))
            for ax, metric in zip(axs, ["return", "cost", "regret"]):
                # Only take the results at the end of training
                filtered_df = df[df["step"] == df["step"].max()]
                # Take the mean over the repetitions
                mean_df = filtered_df.groupby(["Algorithm", "type"]).mean(numeric_only=True)
                
                # Reshape the dataframe to make integration into seaborn easier
                pivot_df = mean_df.reset_index()
                pivot_df = pivot_df.pivot(index='Algorithm', columns='type', values=metric)
                pivot_df = pivot_df.fillna(0)

                # Exclude the outlier that makes reading the other values difficult
                if metric == "regret" and set == "evaluation_" and exclude_outlier:
                    pivot_df = pivot_df[pivot_df["baseline"] < 4000]

                # Only plot legend for the last subplot
                plot_legend = False
                if metric == "regret":
                    plot_legend = True

                # Create the scatterplot
                sns.scatterplot(data=pivot_df, x='baseline', y='curriculum', hue='Algorithm', style='Algorithm',
                                markers=["o"], legend=plot_legend, ax=ax, s=100)
                ax.grid(False)

                # Add labels and ticks
                ax.set_title(f'{metric}')
                ax.set_ylabel('Curriculum')
                ax.set_xlabel('Baseline')

                # Force same range on both axis
                if ax.get_xlim()[1] > ax.get_ylim()[1]:
                    ax.set_xlim(0)
                    ax.set_ylim(0, ax.get_xlim()[1])
                else:
                    ax.set_xlim(0, ax.get_ylim()[1])
                    ax.set_ylim(0)
                ax.set_aspect('equal', adjustable='box')
                
                # Add diagonal
                ax.axline((0, 0), slope=1, color="black", linestyle="--", zorder=1, label='Diagonal')
                # Add PPO lines
                if "non" in set:
                    ax.axvline(pivot_df.loc['PPO', 'baseline'], linestyle="-.", color="#8c564b", label='PPO Train')
                    ax.axhline(pivot_df.loc['PPO', 'baseline'], linestyle="-.", color="#8c564b")
                else:
                    ax.axvline(pivot_df.loc['PPO', 'baseline'], linestyle=":", color="#8c564b", label='PPO Evaluation')
                    ax.axhline(pivot_df.loc['PPO', 'baseline'], linestyle=":", color="#8c564b")

            # Adjust layout and colorbar
            plt.legend(loc=(1.01, 0.01), ncol=1)
            plt.suptitle('Scatterplot of baseline performance vs. curriculum performance')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"./figures/{folder_base}/comparison/{set}comparison{'_outlier' if not exclude_outlier else ''}.png")
            plt.savefig(f"./figures/{folder_base}/comparison/{set}comparison{'_outlier' if not exclude_outlier else ''}.pdf")
            plt.close()

    # Plot non-evaluation metrics
    plot_scatter(train_df, eval_df, exclude_outlier=False)
    plot_scatter(train_df, eval_df)
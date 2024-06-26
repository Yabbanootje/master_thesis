from ...master_thesis.main import *

if __name__ == '__main__':
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 1000
    repetitions = 15
    baseline_algorithms = ["PPO", "PPOLag", "CPO", "FOCOPS", "OnCRPO"] # ["PPO", "PPOLag", "CPO", "FOCOPS", "OnCRPO", "CUP", "PCPO", "PPOEarlyTerminated"]
    curr_algorithms = ["OnCRPO", "CUP", "FOCOPS", "PCPO", "PPOEarlyTerminated", "PPOLag"]
    folder_base = "algorithm_comparison"
    curr_changes = [10, 20, 40, 100]
    seeds = [int(rand.random() * 10000) for i in range(repetitions)]

    # Repeat experiments
    wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
    with Pool(8) as p:
        args_base = list(product(baseline_algorithms, ["baseline"], seeds))
        args_curr = list(product(curr_algorithms, ["curriculum"], seeds))
        args = args_curr + args_base
        p.starmap(use_params, args)

    # Plot the results
    train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
    eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    print_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)

    train_df.to_csv(f"app/figures/{folder_base}/comparison/train_df.csv")
    eval_df.to_csv(f"app/figures/{folder_base}/comparison/eval_df.csv")

    train_df = pd.read_csv(f"app/figures/{folder_base}/comparison/train_df.csv")
    eval_df = pd.read_csv(f"app/figures/{folder_base}/comparison/eval_df.csv")

    def plot_metrics(train_df, eval_df, combine=True, exclude_outlier=True):
        if combine:
            fig, axs = plt.subplots(1, 3, figsize=(13, 5))
            for ax, metric in zip(axs, ["return", "cost", "regret"]):
                filtered_train_df = train_df[train_df["step"] == train_df["step"].max()]
                filtered_eval_df = eval_df[eval_df["step"] == eval_df["step"].max()]
                mean_train_df = filtered_train_df.groupby(["Algorithm", "type"]).mean(numeric_only=True)
                mean_eval_df = filtered_eval_df.groupby(["Algorithm", "type"]).mean(numeric_only=True)
                
                pivot_train_df = mean_train_df.reset_index()
                pivot_train_df = pivot_train_df.pivot(index='Algorithm', columns='type', values=metric)
                pivot_train_df = pivot_train_df.fillna(0)
                pivot_eval_df = mean_eval_df.reset_index()
                pivot_eval_df = pivot_eval_df.pivot(index='Algorithm', columns='type', values=metric)
                pivot_eval_df = pivot_eval_df.fillna(0)

                pivot_train_df["set"] = "Training"
                pivot_eval_df["set"] = "Evaluation"
                pivot_df = pd.concat([pivot_train_df, pivot_eval_df], ignore_index=False)

                if metric == "cost" and exclude_outlier:
                    pivot_df = pivot_df[pivot_df["baseline"] < 100]

                plot_legend = False
                if metric == "regret":
                    plot_legend = True

                sns.scatterplot(data=pivot_df, x='baseline', y='curriculum', hue='Algorithm', style="set", 
                                markers=["o", "$\circ$"], legend=plot_legend, ax=ax, s=100)

                ax.grid(False)

                # Add labels and ticks
                ax.set_title(f'{metric}')
                ax.set_ylabel('Curriculum')
                ax.set_xlabel('Baseline')
                # ax.set_yticks(np.arange(2))

                if ax.get_xlim()[1] > ax.get_ylim()[1]:
                    ax.set_xlim(0)
                    ax.set_ylim(0, ax.get_xlim()[1])
                else:
                    ax.set_xlim(0, ax.get_ylim()[1])
                    ax.set_ylim(0)
                ax.set_aspect('equal', adjustable='box')
                
                ax.axline((0, 0), slope=1, color="black", linestyle="--", zorder=1, label='Diagonal')
                ax.axvline(pivot_train_df.loc['PPO', 'baseline'], linestyle="-.", color="#8c564b", label='PPO Train')
                ax.axhline(pivot_train_df.loc['PPO', 'baseline'], linestyle="-.", color="#8c564b")
                ax.axvline(pivot_eval_df.loc['PPO', 'baseline'], linestyle=":", color="#8c564b", label='PPO Evaluation')
                ax.axhline(pivot_eval_df.loc['PPO', 'baseline'], linestyle=":", color="#8c564b")

            # Adjust layout and colorbarplt.legend(loc=(1.01, 0.01), ncol=1)
            plt.legend(loc=(1.01, 0.01), ncol=1)
            plt.suptitle('Scatterplot of baseline performance vs. curriculum performance')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"app/figures/{folder_base}/comparison/comparison{'_outlier' if not exclude_outlier else ''}.png")
            plt.savefig(f"app/figures/{folder_base}/comparison/comparison{'_outlier' if not exclude_outlier else ''}.pdf")
            plt.show()
            plt.close()

        else:
            for df, set in zip([train_df, eval_df], ["non_evaluation_", "evaluation_"]):
                fig, axs = plt.subplots(1, 3, figsize=(13, 4))
                for ax, metric in zip(axs, ["return", "cost", "regret"]):
                    filtered_df = df[df["step"] == df["step"].max()]
                    mean_df = filtered_df.groupby(["Algorithm", "type"]).mean(numeric_only=True)
                    
                    pivot_df = mean_df.reset_index()
                    pivot_df = pivot_df.pivot(index='Algorithm', columns='type', values=metric)
                    pivot_df = pivot_df.fillna(0)

                    if metric == "cost" and exclude_outlier:
                        pivot_df = pivot_df[pivot_df["baseline"] < 100]
                    if metric == "regret" and set == "evaluation_" and exclude_outlier:
                        pivot_df = pivot_df[pivot_df["baseline"] < 4000]

                    plot_legend = False
                    if metric == "regret":
                        plot_legend = True

                    if "non" in set:
                        markers = ["o"]
                    else:
                        markers = ["$\circ$"]

                    sns.scatterplot(data=pivot_df, x='baseline', y='curriculum', hue='Algorithm', style='Algorithm',
                                    markers=markers, legend=plot_legend, ax=ax, s=100)

                    ax.grid(False)

                    # Add labels and ticks
                    ax.set_title(f'{metric}')
                    ax.set_ylabel('Curriculum')
                    ax.set_xlabel('Baseline')
                    # ax.set_yticks(np.arange(2))

                    if ax.get_xlim()[1] > ax.get_ylim()[1]:
                        ax.set_xlim(0)
                        ax.set_ylim(0, ax.get_xlim()[1])
                    else:
                        ax.set_xlim(0, ax.get_ylim()[1])
                        ax.set_ylim(0)
                    ax.set_aspect('equal', adjustable='box')
                    
                    ax.axline((0, 0), slope=1, color="black", linestyle="--", zorder=1, label='Diagonal')
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
                plt.savefig(f"app/figures/{folder_base}/comparison/{set}comparison{'_outlier' if not exclude_outlier else ''}.png")
                plt.savefig(f"app/figures/{folder_base}/comparison/{set}comparison{'_outlier' if not exclude_outlier else ''}.pdf")
                plt.show()
                plt.close()

    # Plot non-evaluation metrics
    plot_metrics(train_df, eval_df)
    plot_metrics(train_df, eval_df, combine=False)
    plot_metrics(train_df, eval_df, exclude_outlier=False)
    plot_metrics(train_df, eval_df, combine=False, exclude_outlier=False)
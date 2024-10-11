import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *

if __name__ == '__main__':
    folder_base = "grid_search"
    eval_episodes = 3
    cost_limit = 5.0
    safe_freq = 20
    epochs = 100
    repetitions = 3
    baseline_algorithms = ["PPOLag"]
    curr_algorithms = ["PPOLag"]

    # Grid search params
    cost_limits = [5.0]
    lag_multiplier_inits = [0.001, 0.01, 0.1]
    lag_multiplier_lrs = [0.01, 0.035, 0.05, 0.1]
    steps_per_epochs = [1000]
    update_iterss = [1, 10, 50]
    nn_sizes = [64, 256]
    parameters = ["lag_multiplier_inits", "lag_multiplier_lrs", "update_iterss", "nn_sizes"]

    # Train agents
    for grid_params in product(cost_limits, lag_multiplier_inits, lag_multiplier_lrs, steps_per_epochs, update_iterss, nn_sizes):
        (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size) = grid_params
        grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)
        # Create folder
        folder_name = folder_base + "/test-half_curriculum-" + str(grid_params)

        # Repeat experiments
        for i in range(repetitions):
            # Get configurations
            base_cfgs = get_configs(folder=folder_name + "/baseline", algos=baseline_algorithms, epochs=epochs, cost_limit=cost_limit,
                                    steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size,
                                    lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)
            curr_cfgs = get_configs(folder=folder_name + "/curriculum", algos=curr_algorithms, epochs=epochs, cost_limit=cost_limit,
                                    steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size,
                                    lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)

            # Initialize agents
            baseline_env_id = 'SafetyPointHM3-v0'
            curr_env_id = 'SafetyPointHM0-v0'

            baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
            curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

            # Train agents
            for baseline_agent in baseline_agents:
                train_agent(baseline_agent, eval_episodes, True)

            for curriculum_agent in curriculum_agents:
                train_agent(curriculum_agent, eval_episodes, True)


    # Save data to csv
    last_means = pd.DataFrame(columns = parameters + ["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", 
                                                      "Evaluation Regret",
                                                      "Return Curr", "Cost Curr", "Regret Curr", "Evaluation Return Curr",
                                                      "Evaluation Cost Curr", 
                                                      "Evaluation Regret Curr"]
                                                      ).set_index(parameters)
    
    for grid_params in product(cost_limits, lag_multiplier_inits, lag_multiplier_lrs, steps_per_epochs, update_iterss, nn_sizes):
        (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size) = grid_params
        # Create folder
        folder_base = folder_base + "/test-half_curriculum-" + str(grid_params)

        # Plot the results and get data
        curr_changes = [10, 20, 30]
        train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
        eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, save_freq=save_freq)

        # Get values for baseline
        filtered_train_df = train_df[train_df['type'] == "baseline"]
        filtered_eval_df = eval_df[eval_df['type'] == "baseline"]
        mean_train_df = filtered_train_df.groupby(["step"]).mean(numeric_only=True)
        mean_eval_df = filtered_eval_df.groupby(["step"]).mean(numeric_only=True)

        return_ = mean_train_df["return"].iloc[-1]
        cost = mean_train_df["cost"].iloc[-1]
        eval_return = mean_eval_df["return"].iloc[-1]
        eval_cost = mean_eval_df["cost"].iloc[-1]
        eval_length = mean_eval_df["length"].iloc[-1]
        auc_cost = np.trapz(mean_train_df["cost"], dx=1)
        auc_eval_cost = np.trapz(mean_eval_df["cost"], dx=save_freq)
        regret_train = mean_train_df["regret"].iloc[-1]
        regret_eval = mean_eval_df["regret"].iloc[-1]

        # Get values for curriculum
        filtered_train_df_curr = train_df[train_df['type'] == "curriculum"]
        filtered_eval_df_curr = eval_df[eval_df['type'] == "curriculum"]
        mean_train_df_curr = filtered_train_df_curr.groupby(["step"]).mean(numeric_only=True)
        mean_eval_df_curr = filtered_eval_df_curr.groupby(["step"]).mean(numeric_only=True)

        return__curr = mean_train_df_curr["return"].iloc[-1]
        cost_curr = mean_train_df_curr["cost"].iloc[-1]
        eval_return_curr = mean_eval_df_curr["return"].iloc[-1]
        eval_cost_curr = mean_eval_df_curr["cost"].iloc[-1]
        eval_length_curr = mean_eval_df_curr["length"].iloc[-1]
        auc_cost_curr = np.trapz(mean_train_df_curr["cost"], dx=1)
        auc_eval_cost_curr = np.trapz(mean_eval_df_curr["cost"], dx=save_freq)
        regret_train_curr = mean_train_df_curr["regret"].iloc[-1]
        regret_eval_curr = mean_eval_df_curr["regret"].iloc[-1]

        parameter_means = pd.DataFrame(data = {"lag_multiplier_inits": lag_multiplier_init, 
                                "lag_multiplier_lrs": lag_multiplier_lr, "update_iterss": update_iters, "nn_sizes": nn_size, 
                                "Return": return_, "Cost": cost, "Regret": regret_train, 'Evaluation Return': eval_return,
                                'Evaluation Cost': eval_cost, "Evaluation Regret": regret_eval,
                                "Return Curr": return__curr, "Cost Curr": cost_curr, "Regret Curr": regret_train_curr, 
                                'Evaluation Return Curr': eval_return_curr,
                                'Evaluation Cost Curr': eval_cost_curr, "Evaluation Regret Curr": regret_eval_curr,
                                },
                                index = [0]).set_index(parameters)       

        last_means = pd.concat([last_means, parameter_means])


    # Plot full heatmap
    for algo_type in ["baseline", "curriculum"]:
        # Load data
        last_means = pd.read_csv(f"./figures/{folder_base}/last_means.csv").set_index(parameters)

        if algo_type == "baseline":
            last_means = last_means.sort_values(by=["Evaluation Return"])[["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", "Evaluation Regret"]]
        else:
            last_means = last_means.sort_values(by=["Evaluation Return Curr"])[["Return Curr", "Cost Curr", "Regret Curr", 
                                                                                "Evaluation Return Curr", "Evaluation Cost Curr", "Evaluation Regret Curr"]]

        # Get annotation for heatmap
        annotation = last_means.to_numpy()

        # Normalize columns
        for column in last_means.columns:
            if "Cost" in column:
                last_means[column] = np.log(last_means[column] + 1)
            if "Cost" in column or "Regret" in column:
                last_means[column] = -last_means[column]
            last_means[column] = (last_means[column] - last_means[column].min()) / (last_means[column].max() - last_means[column].min())

        # Plotting the heatmap
        fig = plt.figure(figsize=(12, 13))
        ax_img = plt.imshow(last_means.values, cmap='viridis', aspect='auto')
        ax = ax_img.axes
        plt.grid(False)

        # Add labels and ticks
        plt.title('Heatmap of final epoch performance')
        plt.ylabel('Parameter Combinations\n(lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size)')
        plt.xlabel('Metrics')
        plt.yticks(ticks=np.arange(len(last_means.index)), labels=last_means.index, rotation='horizontal')
        plt.xticks(ticks=np.arange(len(last_means.columns)), labels=["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", "Evaluation Regret"])

        # Show colorbar
        plt.colorbar(label='Normalized mean of the performance in the final epoch')

        # Add original values as text
        promising_parameters_curriculum = [
                                        (0.001, 0.1, 1, 64), 
                                        (0.001, 0.01, 1, 64),
                                        (0.01, 0.1, 1, 256),
                                        (0.1, 0.01, 1, 256),
                                        (0.01, 0.01, 1, 256),
                                        (0.001, 0.01, 10, 64),
                                        (0.1, 0.05, 10, 64),
                                        (0.1, 0.05, 50, 64),
                                        (0.01, 0.035, 1, 256),
                                        (0.001, 0.035, 50, 64),
                                        ]
        
        promising_parameters = [(0.1, 0.01, 1, 64),
                                (0.01, 0.01, 1, 64),
                                (0.001, 0.01, 1, 64),
                                (0.1, 0.01, 1, 256),
                                (0.001, 0.01, 1, 256),
                                (0.1, 0.01, 10, 64),
                                (0.01, 0.01, 10, 64),
                                (0.001, 0.01, 10, 64),
                                (0.1, 0.01, 10, 256),
                                ]
        
        # Get indices in heatmap corresponding to the promising parameters
        promising_indices = []
        for index, i in zip(last_means.index, range(len(annotation))):
            if tuple(index[i] for i in range(len(index))) in promising_parameters:
                promising_indices.append(i)

        promising_indices_curriculum = []
        for index, i in zip(last_means.index, range(len(annotation))):
            if tuple(index[i] for i in range(len(index))) in promising_parameters_curriculum:
                promising_indices_curriculum.append(i)

        # Put textual values inside of the heatmap
        for i in range(len(annotation)):
            for j in range(len(annotation[0])):
                plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center')#, color='white')

        # Color the y-axis according to which promising parameters it belongs
        for i in range(len(annotation)):
            if i in promising_indices and i in promising_indices_curriculum:
                ax.get_yticklabels()[i].set_color("red")
            elif i in promising_indices:
                ax.get_yticklabels()[i].set_color("orange")
            elif i in promising_indices_curriculum:
                ax.get_yticklabels()[i].set_color("black")
            else:
                ax.get_yticklabels()[i].set_color("grey")

        plt.tight_layout()
        plt.savefig(f"figures/{folder_base}/{algo_type}_heatmap_log_costs_color_ticks.png")
        plt.savefig(f"figures/{folder_base}/{algo_type}_heatmap_log_costs_color_ticks.pdf")
        plt.close()

    
    # Plot the sorted heatmaps
    def plot_metrics(metrics, filename_prefix):
        fig, axs = plt.subplots(3, 1, figsize=(13, 7))
        last_means = pd.read_csv(f"./figures/{folder_base}/last_means.csv").set_index(parameters)
        
        for ax, metric in zip(axs, metrics):
            ascending = False
            if "Cost" in metric or "Regret" in metric:
                ascending = True
            metric_base = last_means[metric].sort_values(ascending=ascending).reset_index(drop=True)
            metric_curr = last_means[metric + " Curr"].sort_values(ascending=ascending).reset_index(drop=True)

            # Get annotation for heatmap
            metric_last_means = pd.concat([metric_base, metric_curr], axis=1)
            annotation = metric_last_means.T.to_numpy()

            if "Cost" in metric:
                metric_last_means = metric_last_means.applymap(lambda x: np.log(x + 1))
            if "Cost" in metric or "Regret" in metric:
                metric_last_means = metric_last_means.applymap(lambda x: -x)

            metric_last_means = metric_last_means.T

            # Plotting the heatmap
            im = ax.imshow(metric_last_means.values, cmap='viridis', aspect='auto')
            ax.grid(False)

            # Add labels and ticks
            ax.set_title(f'{metric}')
            ax.set_ylabel('Agent type')
            ax.set_xlabel('')
            ax.set_yticks(np.arange(2))
            ax.set_yticklabels(["Baseline", "Curriculum"], rotation='horizontal')

            for i in range(len(annotation)):
                for j in range(len(annotation[0])):
                    ax.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', rotation="vertical")#, color='white')

        # Adjust layout and colorbar
        # fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.1, label='Mean of the performance\nin the final epoch')
        plt.suptitle('Comparison heatmap of final epoch performance')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison.png")
        plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison.pdf")
        plt.close()

    # Define the metric groups
    non_evaluation_metrics = ["Return", "Cost", "Regret"]
    evaluation_metrics = ["Evaluation Return", "Evaluation Cost", "Evaluation Regret"]

    # Plot non-evaluation metrics
    plot_metrics(non_evaluation_metrics, 'non_evaluation')

    # Plot evaluation metrics
    plot_metrics(evaluation_metrics, 'evaluation')


    # Plot the bar charts
    def plot_metrics(metrics, filename_prefix):
        fig, axs = plt.subplots(3, 1, figsize=(13, 7))
        last_means = pd.read_csv(f"./figures/{folder_base}/last_means.csv").set_index(parameters)
        last_means = last_means.sort_values(by='Evaluation Return', ascending=False)
        
        for ax, metric in zip(axs, metrics):
            # Get annotation for heatmap
            print(last_means[[metric, metric + " Curr"]].reset_index(drop=True).head(30))
            metric_last_means = last_means[metric + " Curr"] - last_means[metric]
            metric_last_means = metric_last_means.reset_index(drop=True)
            print(metric_last_means.head(30))

            # Plot the baseline line
            ax.axhline(y=0, color='black', linestyle='-', label='Baseline')

            # Plot the deviation bars
            ax.bar(metric_last_means.index, metric_last_means, label='Deviation', color='skyblue')

            # Add labels and legend
            ax.set_xlabel("Parameter Combination")
            ax.set_ylabel(r"$\Delta$ " + metric)
            ax.set_title(metric)
            ax.set_xlim(-1, 72)
            ax.set_xticks([])

        # Adjust layout and colorbar
        # fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.1, label='Mean of the performance\nin the final epoch')
        plt.suptitle('Relative final epoch performance of curriculum agents')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison_bar.png")
        plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison_bar.pdf")
        plt.close()

    # Plot non-evaluation metrics
    plot_metrics(non_evaluation_metrics, 'non_evaluation')

    # Plot evaluation metrics
    plot_metrics(evaluation_metrics, 'evaluation')
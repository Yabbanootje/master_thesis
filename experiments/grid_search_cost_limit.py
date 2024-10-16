import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *

if __name__ == '__main__':
    folder_base = "grid_search_cost_limit"
    eval_episodes = 3
    cost_limits = [1.0, 5.0]
    steps_per_epoch = 1000
    safe_freq = 20
    epochs = 100
    repetitions = 5
    baseline_algorithms = ["PPOLag"]
    curr_algorithms = ["PPOLag"]

    # Grid search params
    parameters = ["cost_limits", "lag_multiplier_inits", "lag_multiplier_lrs", "update_iterss", "nn_sizes"]

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

    last_means = pd.DataFrame(columns = parameters + ["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", 
                                                      "Evaluation Regret",
                                                      "Return Curr", "Cost Curr", "Regret Curr", "Evaluation Return Curr",
                                                      "Evaluation Cost Curr", 
                                                      "Evaluation Regret Curr"]
                                                      ).set_index(parameters)

    # Traing agents and save data
    for promising_parameter_combo in promising_parameters:
        for cost_limit in cost_limits:
            (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size) = promising_parameter_combo
            grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)
            # Create folder
            folder_name = folder_base + "/test-half_curriculum-" + str(grid_params)

            # Repeat experiments
            for i in range(repetitions):
                # Get configurations
                base_cfgs = get_configs(folder=folder_name + "/baseline", algos=baseline_algorithms, epochs=epochs, cost_limit=cost_limit,
                                        steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size, safe_freq = 20,
                                        lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)
                curr_cfgs = get_configs(folder=folder_name + "/curriculum", algos=curr_algorithms, epochs=epochs, cost_limit=cost_limit,
                                        steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size, safe_freq = 20,
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

            # Plot the results
            curr_changes = [10, 20, 30]
            train_df = plot_train(folder_name, curr_changes, cost_limit, include_weak=False)
            eval_df = plot_eval(folder_name, curr_changes, cost_limit, include_weak=False, save_freq=save_freq)

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

            parameter_means = pd.DataFrame(data = {"cost_limits": cost_limit, "lag_multiplier_inits": lag_multiplier_init, 
                                    "lag_multiplier_lrs": lag_multiplier_lr, "update_iterss": update_iters, "nn_sizes": nn_size, 
                                    "Return": return_, "Cost": cost, "Regret": regret_train, 'Evaluation Return': eval_return,
                                    'Evaluation Cost': eval_cost, "Evaluation Regret": regret_eval,
                                    "Return Curr": return__curr, "Cost Curr": cost_curr, "Regret Curr": regret_train_curr, 
                                    'Evaluation Return Curr': eval_return_curr,
                                    'Evaluation Cost Curr': eval_cost_curr, "Evaluation Regret Curr": regret_eval_curr,
                                    },
                                    index = [0]).set_index(parameters)       

            last_means = pd.concat([last_means, parameter_means])

    last_means.to_csv(f"app/figures/{folder_base}/last_means.csv")


    # Plot heatmap using nine best baseline parameters
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
        plt.title('Heatmap of final epoch performance', size=14)
        plt.ylabel('Parameter Combinations\n(cost_limit, lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size)', size=12)
        plt.xlabel('Metrics', size=12)
        plt.yticks(ticks=np.arange(len(last_means.index)), labels=last_means.index, rotation='horizontal', fontsize=11)
        plt.xticks(ticks=np.arange(len(last_means.columns)), labels=[col.replace(" Curr", "") for col in last_means.columns])

        # Show colorbar
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='Normalized mean of the performance in the final epoch', size=12)
    

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
        promising_indices_curriculum = []
        for index, i in zip(last_means.index, range(len(annotation))):
            if tuple(index[i] for i in range(len(index)) if i != 0) in promising_parameters_curriculum:
                promising_indices_curriculum.append(i)

        promising_indices = []
        for index, i in zip(last_means.index, range(len(annotation))):
            if tuple(index[i] for i in range(len(index)) if i != 0) in promising_parameters:
                promising_indices.append(i)

        # Put textual values inside of the heatmap
        for i in range(len(annotation)):
            for j in range(len(annotation[0])):
                plt.text(j, i, f'{annotation[i, j]:.2f}', fontsize=14, ha='center', va='center')#, color='white')

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
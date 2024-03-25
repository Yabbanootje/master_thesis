from ...master_thesis.main import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost_limit', dest='cost_limit', type=float, help='Add cost_limit', default=5.0)
    args = parser.parse_args()

    eval_episodes = 3
    cost_limits = [1.0, 5.0]
    steps_per_epoch = 1000
    safe_freq = 20
    epochs = 100
    repetitions = 5
    baseline_algorithms = ["PPOLag"] # ["PPO", "PPOLag", "P3O"]
    curr_algorithms = ["PPOLag"] # ["PPOEarlyTerminated", "PPOLag", "CPPOPID", "CPO", "IPO", "P3O"]

    # Grid search params
    parameters = ["cost_limits", "lag_multiplier_inits", "lag_multiplier_lrs", "steps_per_epochs", 
                  "update_iterss", "nn_sizes"]

    promising_parameters_curriculum = [
                                    #    (0.001, 0.1, 1, 64), 
                                       (0.001, 0.01, 1, 64),
                                    #    (0.01, 0.1, 1, 256),
                                    #    (0.1, 0.01, 1, 256),
                                       (0.01, 0.01, 1, 256),
                                       (0.001, 0.01, 10, 64),
                                    #    (0.1, 0.05, 10, 64),
                                    #    (0.1, 0.05, 50, 64),
                                    #    (0.01, 0.035, 1, 256),
                                    #    (0.001, 0.035, 50, 64),
                                       ]
    
    promising_parameters = [(0.1, 0.01, 1, 64), # seems to be one of few that is better with a lower cost limit
                            (0.01, 0.01, 1, 64), # decent when looking at statistics, but when looking at evaluation it is very poor
                            (0.001, 0.01, 1, 64),
                            (0.1, 0.01, 1, 256),
                            (0.001, 0.01, 1, 256),
                            (0.1, 0.01, 10, 64), # seems to be one of few that is better with a lower cost limit
                            # (0.01, 0.01, 10, 64),
                            # (0.001, 0.01, 10, 64),
                            # (0.1, 0.01, 10, 256),
                            ]

    last_means = pd.DataFrame(columns = parameters + ["reward", "cost", "uac_cost", "eval_reward", "eval_cost", 
                                                      "uac_eval_cost", "eval_length"]).set_index(parameters)
    
    for promising_parameter_combo in promising_parameters:
        for cost_limit in cost_limits:
            (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size) = promising_parameter_combo
            grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)
            # Create folder
            folder_name = "grid_search_3/test-half_curriculum-" + str(grid_params)

            # Repeat experiments
            for i in range(repetitions):
                # Get configurations
                base_cfgs = get_configs(folder=folder_name + "/baseline", algos=baseline_algorithms, epochs=epochs, 
                                        cost_limit=cost_limit, random=True, steps_per_epoch = steps_per_epoch, 
                                        update_iters = update_iters, nn_size = nn_size, safe_freq = safe_freq,
                                        lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)
                curr_cfgs = get_configs(folder=folder_name + "/curriculum", algos=curr_algorithms, epochs=epochs, 
                                        cost_limit=cost_limit, random=True, steps_per_epoch = steps_per_epoch, 
                                        update_iters = update_iters, nn_size = nn_size, safe_freq = safe_freq,
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
            means = plot_train(folder_name, curr_changes, cost_limit, include_weak=False, mean_baseline=False)
            eval_means = plot_eval(folder_name, curr_changes, cost_limit, include_weak=False, mean_baseline=False)

            reward = means["rewards"][-1]
            cost = means["costs"][-1]
            eval_reward = eval_means["rewards"][-1]
            eval_cost = eval_means["costs"][-1]
            eval_length = eval_means["lengths"][-1]
            uac_cost = np.trapz(means["costs"], dx=1)
            uac_eval_cost = np.trapz(eval_means["costs"], dx=safe_freq)

            parameter_means = pd.DataFrame(data = {"cost_limits": cost_limit, "lag_multiplier_inits": lag_multiplier_init, "lag_multiplier_lrs": lag_multiplier_lr, 
                                    "steps_per_epochs": steps_per_epoch, "update_iterss": update_iters, "nn_sizes": nn_size, 
                                    "reward": reward, "cost": cost, "uac_cost": uac_cost, 'eval_reward': eval_reward,
                                    'eval_cost': eval_cost, "uac_eval_cost": uac_eval_cost, 'eval_length': eval_length}, 
                                    index = [0]).set_index(parameters)       

            last_means = pd.concat([last_means, parameter_means])

    last_means = last_means.sort_values(by=["eval_reward"])

    # Get annotation for heatmap
    annotation = last_means.to_numpy()

    promising_indices = []
    for index, i in zip(last_means.index, range(len(annotation))):
        if tuple(index[i] for i in range(len(index)) if i not in [0, 3]) in promising_parameters_curriculum:
            promising_indices.append(i)

    # Normalize columns
    for column in last_means.columns:
        if column == "eval_cost" or column == "cost":
            last_means[column] = np.log(last_means[column] + 1)
        if "cost" in column or "length" in column:
            last_means[column] = -last_means[column]
        last_means[column] = (last_means[column] - last_means[column].min()) / (last_means[column].max() - last_means[column].min())

    # Plotting the heatmap
    plt.figure(figsize=(11, 13))
    plt.imshow(last_means.values, cmap='viridis', aspect='auto')

    # Add labels and ticks
    plt.title('Heatmap of final epoch performance')
    plt.ylabel('Parameter Combinations (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size)')
    plt.xlabel('Metrics')
    plt.yticks(ticks=np.arange(len(last_means.index)), labels=last_means.index, rotation='horizontal')
    plt.xticks(ticks=np.arange(len(last_means.columns)), labels=last_means.columns)

    # Show colorbar
    plt.colorbar(label='Normalized mean of the performance in the final epoch')

    # Add original values as text
    for i in range(len(annotation)):
        for j in range(len(annotation[0])):
            if i in promising_indices:
                plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='red')
            else:
                plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(f"app/figures/grid_search_3/curr_heatmap_log_costs.png")
    plt.show()
    plt.close()
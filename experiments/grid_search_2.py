from ...master_thesis.main import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps_per_epoch', dest='steps_per_epoch', type=int, help='Add steps_per_epoch')
    args = parser.parse_args()

    eval_episodes = 3
    cost_limit = 1.0
    epochs = 100
    repetitions = 5
    baseline_algorithms = ["PPOLag"] # ["PPO", "PPOLag", "P3O"]
    curr_algorithms = ["PPOLag"] # ["PPOEarlyTerminated", "PPOLag", "CPPOPID", "CPO", "IPO", "P3O"]

    # Grid search params
    # cost_limits = [1.0] # [1.0, 5.0, 10.0]
    # lag_multiplier_inits = [0.001, 0.01, 0.1] # [0.001, 0.005, 0.01, 0.1]
    # lag_multiplier_lrs = [0.01, 0.035, 0.05, 0.1] # [args.lag_multiplier_lr]
    steps_per_epochs = [args.steps_per_epoch]
    # update_iterss = [1, 10, 50]
    # nn_sizes = [64, 256] # [64, 128, 256]
    parameters = ["cost_limits", "lag_multiplier_inits", "lag_multiplier_lrs", "steps_per_epochs", "update_iterss", "nn_sizes"]

    promising_parameters = [(0.001, 0.1, 1, 64),
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

    last_means = pd.DataFrame(columns = parameters + ["reward", "cost", "eval_reward", "eval_cost", "eval_length"]).set_index(parameters)

    # for grid_params in product(cost_limits, lag_multiplier_inits, lag_multiplier_lrs, steps_per_epochs, update_iterss, nn_sizes):
        # (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size) = grid_params
    for promising_parameter_combo in promising_parameters:
        for steps_per_epoch in steps_per_epochs:
            (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size) = promising_parameter_combo
            grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)
            # Create folder
            folder_name = "grid_search_2/test-half_curriculum-" + str(grid_params)
            # folder_name = folder_name + "---" + str(datetime.datetime.now()).replace(' ', '-')

            # Repeat experiments
            for i in range(repetitions):
                # Get configurations
                base_cfgs = get_configs(folder=folder_name + "/baseline", algos=baseline_algorithms, epochs=epochs, cost_limit=cost_limit, random=True,
                                        steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size, safe_freq = 20,
                                        lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)
                curr_cfgs = get_configs(folder=folder_name + "/curriculum", algos=curr_algorithms, epochs=epochs, cost_limit=cost_limit, random=True,
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
            means = plot_train(folder_name, curr_changes, cost_limit, include_weak=False)
            eval_means = plot_eval(folder_name, curr_changes, cost_limit, include_weak=False)

            reward = means["rewards"][-1]
            cost = means["costs"][-1]
            eval_reward = eval_means["rewards"][-1]
            eval_cost = eval_means["costs"][-1]
            eval_length = eval_means["lengths"][-1]

            parameter_means = pd.DataFrame(data = {"cost_limits": cost_limit, "lag_multiplier_inits": lag_multiplier_init, "lag_multiplier_lrs": lag_multiplier_lr, 
                                    "steps_per_epochs": steps_per_epoch, "update_iterss": update_iters, "nn_sizes": nn_size, 
                                    "reward": reward, "cost": cost, 'eval_reward': eval_reward,'eval_cost': eval_cost, 
                                    'eval_length': eval_length}, 
                                    index = [0]).set_index(parameters)       

            last_means = pd.concat([last_means, parameter_means])

    last_means = last_means.sort_values(by=["eval_reward"])

    # Get annotation for heatmap
    annotation = last_means.to_numpy()

    # Normalize columns
    for column in last_means.columns:
        if column == "eval_cost" or column == "cost":
            last_means[column] = np.log(last_means[column] + 0.1)
        last_means[column] = (last_means[column] - last_means[column].min()) / (last_means[column].max() - last_means[column].min())

    # Plotting the heatmap
    plt.figure(figsize=(8, 12))
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
            plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(f"app/figures/grid_search_2/heatmap_log_costs.png")
    plt.show()
    plt.close()
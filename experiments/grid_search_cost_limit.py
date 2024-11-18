# This file corresponds to the experiments performed in section 6.2.2 of the thesis

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.plot_functions_grid_search import *
from plotting.plot_functions import plot_train, plot_eval
from main import *
from experiments.grid_search import get_results

if __name__ == "__main__":
    folder_base = "grid_search_cost_limit"
    eval_episodes = 3
    steps_per_epoch = 1000
    safe_freq = 20
    epochs = 100
    repetitions = 5
    baseline_algorithms = ["PPOLag"]
    curr_algorithms = ["PPOLag"]

    # Grid search parameters
    cost_limits = [1.0, 5.0]
    parameters = ["cost_limits", "lag_multiplier_inits", "lag_multiplier_lrs", "update_iterss", "nn_sizes"]

    # Best parameters for curriculum agents from grid_search experiments
    promising_parameters_curriculum = [(0.001, 0.1, 1, 64), 
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
    
    # Best parameters for baseline agents from grid_search experiments
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

    # Create dataframe to record the metrics at the end of training
    last_means = pd.DataFrame(columns = parameters + ["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", 
                                                      "Evaluation Regret",
                                                      "Return Curr", "Cost Curr", "Regret Curr", "Evaluation Return Curr",
                                                      "Evaluation Cost Curr", 
                                                      "Evaluation Regret Curr"]
                                                      ).set_index(parameters)

    # Train agents and record results
    for promising_parameter_combo in promising_parameters:
        for cost_limit in cost_limits:
            (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size) = promising_parameter_combo

            # Create folder for the parameter combination
            grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)
            folder_name = folder_base + "/test-half_curriculum-" + str(grid_params)

            # Repeat experiments
            for i in range(repetitions):
                # Get configurations
                base_cfgs = get_configs(folder=folder_name + "/baseline", algos=baseline_algorithms, epochs=epochs, cost_limit=cost_limit,
                                        steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size, safe_freq = safe_freq,
                                        lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)
                curr_cfgs = get_configs(folder=folder_name + "/curriculum", algos=curr_algorithms, epochs=epochs, cost_limit=cost_limit,
                                        steps_per_epoch = steps_per_epoch, update_iters = update_iters, nn_size = nn_size, safe_freq = safe_freq,
                                        lag_multiplier_init = lag_multiplier_init, lag_multiplier_lr = lag_multiplier_lr)

                # Initialize agents
                baseline_env_id = "SafetyPointHM3-v0"
                curr_env_id = "SafetyPointFrom0HM3-v0"

                baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
                curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

                # Train agents
                for baseline_agent in baseline_agents:
                    train_agent(baseline_agent, eval_episodes, True)

                for curriculum_agent in curriculum_agents:
                    train_agent(curriculum_agent, eval_episodes, True)

            # Plot the results
            curr_changes = [10, 20, 30]
            train_df = plot_train(folder_name, curr_changes, cost_limit)
            eval_df = plot_eval(folder_name, curr_changes, cost_limit, save_freq=save_freq)

            # Record results
            parameter_means = get_results(train_df=train_df, eval_df=eval_df, parameters=parameters, 
                                          parameter_values=(cost_limit, lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size))
            last_means = pd.concat([last_means, parameter_means])

    # Save results
    last_means.to_csv(f"./figures/{folder_base}/last_means.csv")

    # Plot heatmap using nine best baseline parameters
    plot_heatmap(folder_base=folder_base, algo_type="baseline", parameters=parameters, promising_parameters=promising_parameters,
                    promising_parameters_curriculum=promising_parameters_curriculum)
    plot_heatmap(folder_base=folder_base, algo_type="curriculum", parameters=parameters, promising_parameters=promising_parameters,
                    promising_parameters_curriculum=promising_parameters_curriculum)
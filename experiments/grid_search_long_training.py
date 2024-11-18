# This file corresponds to the experiments performed in section 6.2.3 of the thesis

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.plot_functions_grid_search import *
from plotting.plot_functions import plot_train, plot_eval
from main import *
from experiments.grid_search import get_results

if __name__ == "__main__":
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    safe_freq = 10
    epochs = 500
    repetitions = 10
    baseline_algorithms = ["PPOLag"]
    curr_algorithms = ["PPOLag"]
    folder_base = "grid_search_long_training/half_curr"
    
    # Grid search parameters
    parameters = ["lag_multiplier_inits", "lag_multiplier_lrs", 
                  "update_iterss", "nn_sizes"]

    # Best parameters for baseline agents from grid_search experiments
    promising_parameters = [(0.1, 0.01, 1, 64),
                            (0.01, 0.01, 1, 64),
                            (0.001, 0.01, 1, 64),
                            (0.1, 0.01, 1, 256),
                            (0.001, 0.01, 1, 256),
                            (0.1, 0.01, 10, 64),
                            ]
    
    last_means = pd.DataFrame(columns = parameters + ["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", 
                                                      "Evaluation Regret",
                                                      "Return Curr", "Cost Curr", "Regret Curr", "Evaluation Return Curr",
                                                      "Evaluation Cost Curr", 
                                                      "Evaluation Regret Curr"]
                                                      ).set_index(parameters)
    
    for promising_parameter_combo in promising_parameters:
        (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size) = promising_parameter_combo
        grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)

        # Create folder
        folder_name = folder_base + "-" + str(grid_params)

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
            baseline_env_id = "SafetyPointHM3-v0"
            curr_env_id = "SafetyPointFrom0HM3-v0"

            baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
            curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

            # Train agents
            for baseline_agent in baseline_agents:
                train_agent(baseline_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

            for curriculum_agent in curriculum_agents:
                train_agent(curriculum_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

        # Plot the results
        curr_changes = [10, 20, 30]
        train_df = plot_train(folder=folder_name, curr_changes=curr_changes, cost_limit=cost_limit)
        eval_df = plot_eval(folder=folder_name, curr_changes=curr_changes, cost_limit=cost_limit, save_freq=save_freq)

        # Record results
        parameter_means = get_results(train_df=train_df, eval_df=eval_df, parameters=parameters, 
                                      parameter_values=(lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size))
        last_means = pd.concat([last_means, parameter_means])

    # Save results
    last_means.to_csv(f"./figures/{folder_base}/last_means.csv")
    
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


    # Plot full heatmap for baseline and curriculum
    plot_heatmap(folder_base=folder_base, algo_type="baseline", parameters=parameters, promising_parameters=promising_parameters,
                    promising_parameters_curriculum=promising_parameters_curriculum, figsize=(12,5))
    

    # Most promising parameters will be trained for longer
    most_promising_parameters = [(0.1, 0.01, 1, 256),
                                 (0.1, 0.01, 10, 64),
                                 ]
    epochs = 800
    folder_base = "grid_search_longer_training/half_curr"
    
    for promising_parameter_combo in promising_parameters:
        (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size) = promising_parameter_combo
        grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)

        # Create folder
        folder_name = folder_base + "-" + str(grid_params)

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
            baseline_env_id = "SafetyPointHM3-v0"
            curr_env_id = "SafetyPointFrom0HM3-v0"

            baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
            curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

            # Train agents
            for baseline_agent in baseline_agents:
                train_agent(baseline_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

            for curriculum_agent in curriculum_agents:
                train_agent(curriculum_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

        # Plot the results
        curr_changes = [10, 20, 30]
        train_df = plot_train(folder=folder_name, curr_changes=curr_changes, cost_limit=cost_limit)
        eval_df = plot_eval(folder=folder_name, curr_changes=curr_changes, cost_limit=cost_limit, save_freq=save_freq)

        # Record results
        parameter_means = get_results(train_df=train_df, eval_df=eval_df, parameters=parameters, 
                                      parameter_values=(lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size))  
        last_means = pd.concat([last_means, parameter_means])

    # Save results
    last_means.to_csv(f"./figures/{folder_base}/last_means.csv")

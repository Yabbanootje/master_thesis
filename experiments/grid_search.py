# This file corresponds to the experiments performed in section 6.2.1 of the thesis

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.plot_functions_grid_search import *
from plotting.plot_functions import plot_train, plot_eval
from main import *

if __name__ == "__main__":
    folder_base = "grid_search"
    eval_episodes = 3
    cost_limit = 5.0
    safe_freq = 20
    epochs = 100
    repetitions = 3
    baseline_algorithms = ["PPOLag"]
    curr_algorithms = ["PPOLag"]

    # Grid search parameters
    cost_limits = [5.0]
    lag_multiplier_inits = [0.001, 0.01, 0.1]
    lag_multiplier_lrs = [0.01, 0.035, 0.05, 0.1]
    steps_per_epochs = [1000]
    update_iterss = [1, 10, 50]
    nn_sizes = [64, 256]
    parameters = ["lag_multiplier_inits", "lag_multiplier_lrs", "update_iterss", "nn_sizes"]

    # Create dataframe to record the metrics at the end of training
    last_means = pd.DataFrame(columns = parameters + ["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", 
                                                      "Evaluation Regret",
                                                      "Return Curr", "Cost Curr", "Regret Curr", "Evaluation Return Curr",
                                                      "Evaluation Cost Curr", 
                                                      "Evaluation Regret Curr"]
                                                      ).set_index(parameters)
    
    # Function to retreive necessary results from the complete results dataframes
    def get_results(train_df, eval_df, parameters, parameter_values):
        # Get values for baseline
        filtered_train_df = train_df[train_df["type"] == "baseline"]
        filtered_eval_df = eval_df[eval_df["type"] == "baseline"]
        mean_train_df = filtered_train_df.groupby(["step"]).mean(numeric_only=True)
        mean_eval_df = filtered_eval_df.groupby(["step"]).mean(numeric_only=True)

        return_ = mean_train_df["return"].iloc[-1]
        cost = mean_train_df["cost"].iloc[-1]
        eval_return = mean_eval_df["return"].iloc[-1]
        eval_cost = mean_eval_df["cost"].iloc[-1]
        regret_train = mean_train_df["regret"].iloc[-1]
        regret_eval = mean_eval_df["regret"].iloc[-1]

        # Get values for curriculum
        filtered_train_df_curr = train_df[train_df["type"] == "curriculum"]
        filtered_eval_df_curr = eval_df[eval_df["type"] == "curriculum"]
        mean_train_df_curr = filtered_train_df_curr.groupby(["step"]).mean(numeric_only=True)
        mean_eval_df_curr = filtered_eval_df_curr.groupby(["step"]).mean(numeric_only=True)

        return__curr = mean_train_df_curr["return"].iloc[-1]
        cost_curr = mean_train_df_curr["cost"].iloc[-1]
        eval_return_curr = mean_eval_df_curr["return"].iloc[-1]
        eval_cost_curr = mean_eval_df_curr["cost"].iloc[-1]
        regret_train_curr = mean_train_df_curr["regret"].iloc[-1]
        regret_eval_curr = mean_eval_df_curr["regret"].iloc[-1]

        # Record results
        parameter_dict = {}
        for parameter, parameter_value in zip(parameters, parameter_values):
            parameter_dict[parameter] = parameter_value
        parameter_means = pd.DataFrame(data = {"Return": return_, "Cost": cost, "Regret": regret_train, 
                                "Evaluation Return": eval_return, "Evaluation Cost": eval_cost, 
                                "Evaluation Regret": regret_eval, "Return Curr": return__curr, 
                                "Cost Curr": cost_curr, "Regret Curr": regret_train_curr, 
                                "Evaluation Return Curr": eval_return_curr,
                                "Evaluation Cost Curr": eval_cost_curr, "Evaluation Regret Curr": regret_eval_curr,
                                }.update(parameter_dict),
                                index = [0]).set_index(parameters)
        return parameter_means     
    
    # Train agents and record results
    for grid_params in product(cost_limits, lag_multiplier_inits, lag_multiplier_lrs, steps_per_epochs, update_iterss, nn_sizes):
        (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size) = grid_params

        # Create folder for the parameter combination
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

        # Plot the results and get data
        curr_changes = [10, 20, 30]
        train_df = plot_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
        eval_df = plot_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, save_freq=save_freq)

        # Record results
        parameter_means = get_results(train_df=train_df, eval_df=eval_df, parameters=parameters, 
                                      parameter_values=(lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size))
        last_means = pd.concat([last_means, parameter_means])

    # Save results
    last_means.to_csv(f"./figures/{folder_base}/last_means.csv")
    
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
                    promising_parameters_curriculum=promising_parameters_curriculum)
    plot_heatmap(folder_base=folder_base, algo_type="curriculum", parameters=parameters, promising_parameters=promising_parameters,
                    promising_parameters_curriculum=promising_parameters_curriculum)


    # Define the metric groups
    non_evaluation_metrics = ["Return", "Cost", "Regret"]
    evaluation_metrics = ["Evaluation Return", "Evaluation Cost", "Evaluation Regret"]

    # Plot sorted heatmap for all metrics
    plot_sorted_heatmap(folder_base=folder_base, parameters=parameters, metrics=non_evaluation_metrics, 
                        filename_prefix="non_evaluation")
    plot_sorted_heatmap(folder_base=folder_base, parameters=parameters, metrics=evaluation_metrics, 
                        filename_prefix="evaluation")
    
    # Plot bar chart for all metrics
    plot_bar_chart(folder_base=folder_base, parameters=parameters, metrics=non_evaluation_metrics, 
                        filename_prefix="non_evaluation")
    plot_bar_chart(folder_base=folder_base, parameters=parameters, metrics=evaluation_metrics, 
                        filename_prefix="evaluation")
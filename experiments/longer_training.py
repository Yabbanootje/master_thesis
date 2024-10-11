import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *

def save_profile():
    with open('app/test/my_job_output_231165.txt', 'r') as file:
        lines = file.readlines()

    # Extract data from each line
    data = []
    for line in lines:
        line = line.strip()
        if line:  # Check if the line is not empty
            parts = line.split()
            if parts[0] != "ncalls":
                ncalls = parts[0].split("/")[0]
                tottime = float(parts[1])
                percall = float(parts[2])
                cumtime = float(parts[3])
                percall_1 = float(parts[4])
                function = ' '.join(parts[5:])
                data.append([ncalls, tottime, percall, cumtime, percall_1, function])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['ncalls', 'tottime', 'percall', 'cumtime', 'percall.1', 'filename:lineno(function)'])
    df = df.sort_values("cumtime", ascending=False).reset_index()

    df.to_csv("app/test/test.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', dest='experiment', type=int, help='Choose experiment')
    args = parser.parse_args()

    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    safe_freq = 10
    epochs = 800
    repetitions = 10
    baseline_algorithms = ["PPOLag"] # ["PPO", "PPOLag", "P3O"]
    curr_algorithms = ["PPOLag"] # ["PPOEarlyTerminated", "PPOLag", "CPPOPID", "CPO", "IPO", "P3O"]
    folder_base = "longer_training/half_curr"

    # Grid search params
    parameters = ["cost_limits", "lag_multiplier_inits", "lag_multiplier_lrs", "steps_per_epochs", 
                  "update_iterss", "nn_sizes"]
    
    promising_parameters = [
                            # (0.1, 0.01, 1, 64),
                            # (0.01, 0.01, 1, 64),
                            # (0.001, 0.01, 1, 64),
                            (0.1, 0.01, 1, 256),
                            # (0.001, 0.01, 1, 256),
                            (0.1, 0.01, 10, 64),
                            ]

    if args.experiment == 1:
        promising_parameters = promising_parameters[:1]
    elif args.experiment == 2:
        promising_parameters = promising_parameters[1:]
    
    last_means = pd.DataFrame(columns = parameters + ["Return", "Cost", "Regret", "Evaluation Return", "Evaluation Cost", 
                                                      "Evaluation Regret",
                                                      "Return Curr", "Cost Curr", "Regret Curr", "Evaluation Return Curr",
                                                      "Evaluation Cost Curr", 
                                                      "Evaluation Regret Curr"] # , "eval_length"
                                                      ).set_index(parameters)
    
    for promising_parameter_combo in promising_parameters:
        (lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size) = promising_parameter_combo
        grid_params = (cost_limit, lag_multiplier_init, lag_multiplier_lr, steps_per_epoch, update_iters, nn_size)
        # Create folder
        folder_name = folder_base + "-" + str(grid_params)

        # threads = []

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
                train_agent(baseline_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

            for curriculum_agent in curriculum_agents:
                train_agent(curriculum_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

        # Plot the results
        curr_changes = [10, 20, 30]
        train_df = plot_train(folder=folder_name, curr_changes=curr_changes, cost_limit=cost_limit, include_weak=False)
        eval_df = plot_eval(folder=folder_name, curr_changes=curr_changes, cost_limit=cost_limit, save_freq=save_freq)

        filtered_train_df = train_df[train_df['type'] == "baseline"] # "curriculum"
        filtered_eval_df = eval_df[eval_df['type'] == "baseline"] # "curriculum"
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
                                }, #, 'eval_length': eval_length
                                index = [0]).set_index(parameters)    

        last_means = pd.concat([last_means, parameter_means])

    last_means.to_csv("app/figures/longer_training/last_means.csv")

    # Load data
    last_means = pd.read_csv("app/figures/longer_training/last_means.csv").set_index(parameters)

    last_means = last_means[["Return Curr", "Cost Curr", "Regret Curr", "Evaluation Return Curr", "Evaluation Cost Curr", "Evaluation Regret Curr"]]
    last_means = last_means.sort_values(by=["Evaluation Return Curr"])
    
    # Get annotation for heatmap
    annotation = last_means.to_numpy()

    # Normalize columns
    for column in last_means.columns:
        if "Cost" in column:
            last_means[column] = np.log(last_means[column] + 1)
        if "Cost" in column or "length" in column or "Regret" in column:
            last_means[column] = -last_means[column]
        last_means[column] = (last_means[column] - last_means[column].min()) / (last_means[column].max() - last_means[column].min())

    # Plotting the heatmap
    plt.figure(figsize=(12, 5)) # figsize=(11, 5)
    plt.imshow(last_means.values, cmap='viridis', aspect='auto')
    plt.grid(False)

    # Add labels and ticks
    plt.title('Heatmap of final epoch performance')
    # plt.title('Performance heatmap of the top 20 baseline agents based on evaluation return')
    plt.ylabel('Parameter Combinations\n(lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size)')
    plt.xlabel('Metrics')
    plt.yticks(ticks=np.arange(len(last_means.index)), labels=last_means.index, rotation='horizontal')
    plt.xticks(ticks=np.arange(len(last_means.columns)), labels=[col.replace(" Curr", "") for col in last_means.columns])

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
    
    promising_indices_curriculum = []
    for index, i in zip(last_means.index, range(len(annotation))):
        if tuple(index[i] for i in range(len(index))) in promising_parameters_curriculum:
            promising_indices_curriculum.append(i)

    promising_indices = []
    for index, i in zip(last_means.index, range(len(annotation))):
        if tuple(index[i] for i in range(len(index))) in promising_parameters:
            promising_indices.append(i)

    for i in range(len(annotation)):
        for j in range(len(annotation[0])):
            if i in promising_indices and i in promising_indices_curriculum:
                plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='red')
            elif i in promising_indices:
                plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='orange')
            elif i in promising_indices_curriculum:
                plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='black')
            else:
                plt.text(j, i, f'{annotation[i, j]:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(f"app/figures/longer_training/curriculum_heatmap_log_costs_more_colors.png")
    plt.show()
    plt.close()
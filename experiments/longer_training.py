from ...master_thesis.main import *

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
            baseline_env_id = 'SafetyPointHM3-v0'
            curr_env_id = 'SafetyPointHM0-v0'

            baseline_agents = get_agents(baseline_algorithms, baseline_env_id, base_cfgs)
            curriculum_agents = get_agents(curr_algorithms, curr_env_id, curr_cfgs)

            # # Train agents
            for baseline_agent in baseline_agents:
                train_agent(baseline_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

            for curriculum_agent in curriculum_agents:
                train_agent(curriculum_agent, eval_episodes, render_episodes, True, [int(epochs/2), epochs])

        # Plot the results
        curr_changes = [10, 20, 30]
        means_baseline, means_curr = plot_train(folder_name, curr_changes, cost_limit, repetitions, include_weak=False)
        eval_means_baseline, eval_means_curr = plot_eval(folder_name, curr_changes, cost_limit, repetitions, eval_episodes)
        print_eval(folder_name, means_baseline, means_curr, eval_means_baseline, eval_means_curr)
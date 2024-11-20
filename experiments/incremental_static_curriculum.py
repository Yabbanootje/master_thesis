# This file corresponds to the experiments performed in section 6.3.2 up to and including
# section 6.3.5 of the thesis
# Note that the results from the thesis cannot be replicated exactly with this file,
# as throughout the different experiments here, improvements to the curriculum (such as 
# the learning rate reset) have been made and there is no way to opt-out of these 
# improvements

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.plot_functions_incremental import plot_incremental_train, plot_incremental_eval, print_incremental_results
from main import *
import pandas as pd

if __name__ == '__main__':
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 2000
    repetitions = 10
    baseline_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
    curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    folder_base = "incremental_static_curriculum"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [int(rand.random() * 10000) for i in range(repetitions)]

    # Repeat experiments
    for end_task in range(1, len(curr_changes) + 1):
        with Pool(8) as p:
            args_base = list(product([folder_base], [cost_limit], seeds, [save_freq], [epochs], baseline_algorithms, ["baseline"], 
                                    [curr_changes], [eval_episodes], [render_episodes], [end_task], [end_task]))
            args_curr = list(product([folder_base], [cost_limit], seeds, [save_freq], [epochs], curr_algorithms, ["curriculum"], 
                                    [curr_changes], [eval_episodes], [render_episodes], [end_task], [end_task]))
            p.starmap(run_experiment, args_curr + args_base)


    # Plot the results
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    print_incremental_results(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq)

    # Save results
    train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    
    
    # Get figures for ablation study from section 6.3.3
    folder_base = "incremental_static_curriculum_r"

    # Plot the results
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=train_df)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=eval_df)

    # Save results
    train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")


    # Get figures for ablation study from section 6.3.4
    folder_base = "incremental_static_curriculum_ablation"

    # Plot the results
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)

    # Save results
    train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")


    # Get figures for ablation study from section 6.3.5
    folder_base = "incremental_static_curriculum_lag_init"

    # Plot the results
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=train_df)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=eval_df)

    # Save results
    train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    
# This file corresponds to the experiments performed in section 6.4.2 of the thesis

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting.plot_functions import plot_train, plot_eval, print_eval
from plotting.plot_functions_incremental import *
from main import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 3000
    repetitions = 10
    baseline_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
    curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    adapt_curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    folder_base = "incremental_adaptive_curriculum"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [5905, 7337, 572, 5689, 3968, 175, 4678, 9733, 3743, 7596]

    # Repeat experiments
    for end_task in range(0, len(curr_changes) + 1):
        with Pool(8) as p:
            args_curr = list(product(curr_algorithms, [end_task], ["curriculum"], seeds, [exp], [1.0], [20]))
            args_adapt_curr = list(product(curr_algorithms, [end_task], ["adaptive_curriculum"], seeds, [exp], [1.0], [20]))
            p.starmap(use_params, args_curr + args_adapt_curr)


    # Plot the results
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    print_incremental_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)


    # Create a figure that shows the task progression of the four algorithms (Figure 6.15)
    fig = plt.figure(figsize=(18, 6), dpi=200)
    additional_folder = ""
    additional_file_text = ""

    # Use the results that trained up to the target task
    dataframes = [train_df[(train_df["end_task"] == "T") & (train_df["type"] == "adaptive curriculum")], 
                  eval_df[(eval_df["end_task"] == "T") & (eval_df["type"] == "adaptive curriculum")]]

    # Create a FacetGrid
    sns.set_style("whitegrid")
    g = sns.FacetGrid(dataframes[0], hue='Algorithm', col="Algorithm", col_wrap=4, height=4)  # col_wrap controls how many plots per row

    # Map the lineplot onto the FacetGrid
    g.map_dataframe(sns.lineplot, x='step', y="current_task", errorbar="sd" if False else "se", estimator=None, units='seed', alpha=0.4)

    # Set labels and titles
    g.set_axis_labels("x1000 Steps", "Current Task", fontsize=14)
    g.set_titles("{col_name}", size=14)

    # Force the x-axis ticks and labels to show on all plots
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    # Save the plot
    fig.suptitle("Performance of adaptive agents using different betas and kappas")
    plt.tight_layout(pad=2)
    if not os.path.isdir(f"figures/{folder_base}/{additional_folder}"):
        os.makedirs(f"figures/{folder_base}/{additional_folder}")
    plt.savefig(f"figures/{folder_base}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}current_task.png")
    plt.savefig(f"figures/{folder_base}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}current_task.pdf")
    plt.close()
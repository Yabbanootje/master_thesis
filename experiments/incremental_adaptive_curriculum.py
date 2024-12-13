# This file corresponds to the experiments performed in section 6.4.2 of the thesis

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            args_curr = list(product([folder_base], [cost_limit], seeds, [save_freq], [epochs], curr_algorithms, ["curriculum"], 
                                    [curr_changes], [eval_episodes], [render_episodes], [end_task]))
            args_adapt_curr = list(product([folder_base], [cost_limit], seeds, [save_freq], [epochs], adapt_curr_algorithms, 
                                     ["adaptive_curriculum"], [curr_changes], [eval_episodes], [render_episodes], [end_task]))
            p.starmap(run_experiment, args_curr + args_adapt_curr)


    # Plot the results
    # train_df = pd.read_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    # eval_df = pd.read_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit)
    print_incremental_results(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq)

    # Save results
    train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")


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


    # Function that creates a grid using the last three end tasks and four metrics
    def create_subplot_grid_3_by_4(train_df, eval_df, curr_changes, additional_folder="", additional_file_text=""):
        end_tasks = train_df['end_task'].unique()
        
        # Create a 3x4 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 6.4), dpi=200)
        for ax_row, end_task in zip(axes, end_tasks[-3:]):
            for ax, metric in zip(ax_row, ["return", "success", "cost", "regret"]):
                sns.set_style("whitegrid")

                # Use evaluation data only for the success rate
                if metric == "success":
                    combined_df = eval_df
                else:
                    combined_df = train_df
                
                # Plot the line for this end task and metric
                sns.lineplot(data=combined_df[combined_df['end_task'] == end_task], x='step', y=metric, hue='type', errorbar="se", ax=ax)
                
                # Plot the epochs at which a task change occurs
                if end_task == "T":
                    idx = 6
                else:
                    idx = int(end_task)
                for change in curr_changes[:idx]:
                    ax.axvline(x=change, color="gray", linestyle='-')
                
                # Plot the cost limit
                if metric == 'cost':
                    ax.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')
                    
                # Create titles, labels and legend
                ax.set_xlabel("x1000 Steps")
                ax.get_legend().remove()
                if end_task == "4":
                    ax.set_title(metric.replace('_', ' ').capitalize(), fontsize=14)
                if metric == "return":
                    ax.set_ylabel(f"Task {end_task}", rotation=0, loc="top", fontsize=14)
                else:
                    ax.set_ylabel('')
                if metric == "return" and end_task == "4":
                    handles, labels = ax.get_legend_handles_labels()
                    fig.subplots_adjust(top=0.3)
                    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=14)
        
        # Save the plot
        plt.tight_layout(pad=2, rect=[0, 0, 1, 0.97])
        if not os.path.isdir(f"figures/{folder_base}/{additional_folder}"):
            os.makedirs(f"figures/{folder_base}/{additional_folder}")
        plt.savefig(f"figures/{folder_base}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.png")
        plt.savefig(f"figures/{folder_base}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.pdf")
        plt.close()


    # Exclude results from task 0
    train_df = train_df[train_df["end_task"] != "0"]
    eval_df = eval_df[eval_df["end_task"] != "0"]

    # Remove underscore for figures
    train_df["type"] = train_df["type"].replace("adaptive_curriculum", "adaptive curriculum")
    eval_df["type"] = eval_df["type"].replace("adaptive_curriculum", "adaptive curriculum")

    train_df = train_df.astype({"end_task": str})
    eval_df = eval_df.astype({"end_task": str})

    # Call the function to create the grid of plots
    create_subplot_grid_3_by_4(train_df=train_df[train_df["Algorithm"] == "PPOLag"], 
                               eval_df=eval_df[eval_df["Algorithm"] == "PPOLag"], curr_changes=curr_changes, 
                               additional_folder="PPOLag", additional_file_text="return_success_")

import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def plot_incremental_train(folder, curr_changes, cost_limit, combined_df=None, include_seeds=False, use_std=False):
    # Get folder names for all algorithms
    baseline_dir = "results/" + folder + "/baseline"
    curr_dir = "results/" + folder + "/curriculum"
    adapt_curr_dir = f"results/" + folder + "/adaptive_curriculum"

    # Function to read progress csv and concatenate results into a dataframe
    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            for path in paths:
                # For each repetition of the algorithm
                path = path.replace("\\", "/")

                if "HM0" in path or "HMA0" in path or "HMR0" in path:
                    # Do not save results from task 0
                    break

                # Add current_task if available
                if "curriculum" in algorithm_type:
                    df = pd.read_csv(os.path.join(path, "progress.csv")).rename(columns=
                        {"Metrics/EpRet": "return", "Metrics/EpCost": "cost", "Metrics/EpLen": "length", "Current_task": "current_task"}
                    )[['return', 'cost', 'length', 'current_task']]
                else:
                    df = pd.read_csv(os.path.join(path, "progress.csv")).rename(columns=
                        {"Metrics/EpRet": "return", "Metrics/EpCost": "cost", "Metrics/EpLen": "length"}
                    )[['return', 'cost', 'length']]

                df['Algorithm'] = algorithm.split("-")[0]
                end_version_pattern = r'HMA?R?(\d+|T)'
                end_version = re.search(end_version_pattern, algorithm.split("-")[1])
                df['end_task'] = str(end_version.group(1))
                df['type'] = algorithm_type
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-1].split("-")[1]
                df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)
                df = df.sort_index()
                df['regret'] = df['regret_per_epoch'].cumsum()
                dfs.append(df)
        return pd.concat(dfs)

    # If the data is given, skip the acquisition part
    if combined_df is not None:
        combined_df = combined_df
    else:
        dfs = []
        if os.path.isdir(baseline_dir):
            baseline_df = read_and_concat(baseline_dir, os.listdir(baseline_dir), 'baseline')
            dfs.append(baseline_df)
        if os.path.isdir(curr_dir):
            curr_df = read_and_concat(curr_dir, os.listdir(curr_dir), 'curriculum')
            dfs.append(curr_df)
        if os.path.isdir(adapt_curr_dir):
            adapt_curr_df = read_and_concat(adapt_curr_dir, os.listdir(adapt_curr_dir), 'adaptive_curriculum')
            dfs.append(adapt_curr_df)

        # Combine the dataframes
        combined_df = pd.concat(dfs).reset_index(names="step")

        # Exclude results from task 0
        combined_df = combined_df[combined_df["end_task"] != "0"]

        # Remove underscore for figures
        combined_df["type"] = combined_df["type"].replace("adaptive_curriculum", "adaptive curriculum")
    
    # Create figures folder
    if not os.path.isdir("figures/" + folder):
        os.makedirs("figures/" + folder)

    # Function that creates standard line plots for several metrics
    def create_plot(combined_df, curr_changes=curr_changes, additional_folder = "", additional_file_text = "", additional_title_text = ""):
        for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret']:
            # Plotting using Seaborn
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5), dpi=200)

            # Include a zoomed in cost curve
            zoomed = ""
            if metric == 'cost_zoom':
                metric = 'cost'
                plt.ylim(0, 2 * cost_limit)
                zoomed = "_zoom"

            # Plot the lines
            sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
            if include_seeds:
                ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', estimator=None, legend=False)

            # Plot the epochs at which a task change occurs
            for change in curr_changes:
                plt.axvline(x=change, color="gray", linestyle='-')

            # Plot the cost limit
            if metric == 'cost':
                plt.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

            # Save the plot
            plt.legend(loc=(1.01, 0.01), ncol=1)
            if include_seeds:
                plt.setp(ax.lines[2:], alpha=0.2)
            plt.tight_layout(pad=2)
            plt.title(f"{metric.replace('_', ' ').capitalize()}s of{' ' + additional_title_text if additional_title_text != '' else ''} agents using curriculum and baseline agent")
            plt.xlabel("x1000 Steps")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
                os.makedirs(f"figures/{folder}/{additional_folder}")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}.png")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}.pdf")
            plt.close()

    # Create plots for whole data
    create_plot(combined_df=combined_df)

    # Create plots for each algorithm
    for algo in combined_df["Algorithm"].unique():
        create_plot(combined_df=combined_df[combined_df['Algorithm'] == algo], additional_folder=algo, additional_title_text=algo)

    # Create plots for each environment
    for end_task in combined_df["end_task"].unique():
        if end_task == "T":
            idx = 6
        else:
            idx = int(end_task)
        create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], curr_changes=curr_changes[:idx], additional_folder="HM" + str(end_task), 
                    additional_title_text="HM" + str(end_task))

    # Function that creates a grid using the last three end tasks and the three main metrics
    def create_subplot_grid_3_by_3(combined_df, curr_changes, additional_folder="", additional_file_text=""):
        end_tasks = combined_df['end_task'].unique()
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 6.4), dpi=200)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            for ax, end_task in zip(ax_row, end_tasks[-3:]):
                sns.set_style("whitegrid")
                
                # Plot the line for this end task and metric
                sns.lineplot(data=combined_df[combined_df['end_task'] == end_task], x='step', y=metric, hue='type', errorbar="sd" if use_std else "se", ax=ax)
                
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
                    ax.set_ylabel(metric.replace('_', ' ').capitalize())
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}")
                if metric == "cost" and end_task == "T":
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=(1.01, 0.01), ncol=1)
        
        # Save the plot
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subplot_grid_3_by_3(combined_df=combined_df, curr_changes=curr_changes)
    create_subplot_grid_3_by_3(combined_df=combined_df[combined_df["seed"].isin(map(str, [5905, 7337, 572, 5689, 3968]))], curr_changes=curr_changes, additional_file_text="fair_")
        
    # Function that creates a grid using the all end tasks and the three main metrics
    def create_subplot_grid_6_by_3(combined_df, curr_changes, additional_folder="", additional_file_text=""):
        end_tasks = combined_df['end_task'].unique()
        
        # Create a 6x3 subplot grid
        fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(12, 12.8), dpi=200)
        for ax_row, end_task in zip(axes, end_tasks):
            for ax, metric in zip(ax_row, ["return", "cost", "regret"]):
                sns.set_style("whitegrid")
                
                # Plot the line for this end task and metric
                sns.lineplot(data=combined_df[combined_df['end_task'] == end_task], x='step', y=metric, hue='type', errorbar="sd" if use_std else "se", ax=ax)
                
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
                if end_task == "1":
                    ax.set_title(metric.replace('_', ' ').capitalize(), fontsize=14)
                if metric == "return":
                    ax.set_ylabel(f"Task {end_task}", loc="top", rotation=0, fontsize=14)
                else:
                    ax.set_ylabel('')
                if metric == "cost" and end_task == "T":
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=(0.01, -1.2), ncol=1)
        
        # Save the plot
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subplot_grid_6_by_3(combined_df=combined_df[combined_df["Algorithm"] == "PPOLag"], curr_changes=curr_changes, additional_folder="PPOLag")

    # Function that creates a grid of spiderplots using the all end tasks and the three main metrics
    def create_subspiderplot_grid(combined_df, curr_changes, additional_folder="", additional_file_text=""):
        end_tasks = combined_df['end_task'].unique()
        theta = radar_factory(4, frame='polygon')
        algorithms = ["FOCOPS", "CUP", "PPOEarlyTerminated", "PPOLag"]
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(21, 10), dpi=200, subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=-0.85, hspace=-0.95)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            for ax, end_task in zip(ax_row, end_tasks):
                # Plot PPO
                filtered_df = combined_df[(combined_df["Algorithm"] == "PPO") & (combined_df['end_task'] == end_task)]
                mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                ppo_metrics = [mean_df[metric].iloc[-1] for _ in range(len(algorithms))]
                ax.plot(theta, ppo_metrics, color="red")

                # Plot the main algorithms
                for algorithm_type in combined_df["type"].unique():
                    metrics = []
                    for algorithm in algorithms:
                        filtered_df = combined_df[(combined_df["Algorithm"] == algorithm) & (combined_df['type'] == algorithm_type) & (combined_df['end_task'] == end_task)]
                        mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                        metrics.append(mean_df[metric].iloc[-1])
                    ax.plot(theta, metrics)
                    ax.fill(theta, metrics, alpha=0.15, label='_nolegend_')

                # Set labels with hard-coded spacing
                ax.set_varlabels(["FOCOPS", "CUP   ", "PPOEarlyTerminated", "     PPOLag"], fontsize=13)
                    
                # Create titles and labels
                ax.set_xlabel("x1000 Steps")
                if end_task == "1":
                    ax.set_ylabel(metric.replace('_', ' ').capitalize(), loc="top", rotation=0, fontsize=18)
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}", fontsize=18)

        # Create legend
        labels = ("PPO", "baseline", "curriculum", "adaptive\ncurriculum")
        legend = axes[0, 0].legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize=14)
    
        # Save the plot
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subspiderplot_grid(combined_df=combined_df[combined_df["Algorithm"].isin(["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO"])], 
                              curr_changes=curr_changes, additional_file_text="new_")

    return combined_df

def plot_incremental_eval(folder, curr_changes, cost_limit, combined_df=None, include_seeds=False, include_repetitions=False, use_std=False):
    # Get folder names for all algorithms
    baseline_dir = "results/" + folder + "/baseline"
    curr_dir = "results/" + folder + "/curriculum"
    adapt_curr_dir = f"results/" + folder + "/adaptive_curriculum"

    # Function that extracts values from the logs
    def extract_values(pattern, text):
        return [float(match.group(1)) for match in re.finditer(pattern, text)]

    # Function to read progress csv and concatenate results into a dataframe
    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            seed_paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            eval_paths = [os.path.join(path, "evaluation") for path in seed_paths]

            for path in eval_paths:
                # For each repetition of the algorithm
                path = path.replace("\\", "/")
                
                if "HM0" in path or "HMA0" in path:
                    # Do not save results from task 0
                    break

                epochs = [entry.name for entry in os.scandir(path)]

                returns = []
                costs = []
                lengths = []
                steps = []

                # Variable to hold the number of evaluation episodes
                reps = 0

                for epoch in epochs:
                    # For each epoch at which the agent was evaluated
                    with open(os.path.join(path, epoch, "result.txt"), 'r') as file:
                        # Read the results from the logs
                        data = file.read()

                        # Read results of all of the evaluation episodes into lists
                        return_ = extract_values(r'Episode reward: ([\d\.-]+)', data)
                        cost_ = extract_values(r'Episode cost: ([\d\.-]+)', data)
                        length_ = extract_values(r'Episode length: ([\d\.-]+)', data)

                        # Collect all results over epochs in a single list per metric
                        returns += return_
                        costs += cost_
                        lengths += length_

                        # Get the number of evaluation episodes
                        if reps == 0:
                            reps = len(return_)

                        # Associate each episode result with the same epoch step
                        index = int(epoch.split("-")[1])
                        steps += [index for i in range(reps)]

                # Save all results in a dataframe
                df = pd.DataFrame({'return': returns, 'cost': costs, 'length': lengths, 'step': steps})
                df['Algorithm'] = algorithm.split("-")[0]
                end_version_pattern = r'HMR?A?(\d+|T)'
                end_version = re.search(end_version_pattern, algorithm.split("-")[1])
                df['end_task'] = str(end_version.group(1))
                df['type'] = algorithm_type
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-2].split("-")[1]
                df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)
                df['step'] = pd.to_numeric(df['step'])
                df = df.sort_values(by=['step']).reset_index()
                avg_regret_per_epoch= df.groupby(df.index // reps)['regret_per_epoch'].mean()
                df['regret'] = avg_regret_per_epoch.cumsum().repeat(reps).reset_index(drop=True)
                dfs.append(df)
        return pd.concat(dfs)

    # If the data is given, skip the acquisition part
    if combined_df is not None:
        combined_df = combined_df
    else:
        dfs = []
        if os.path.isdir(baseline_dir):
            baseline_df = read_and_concat(baseline_dir, os.listdir(baseline_dir), 'baseline')
            dfs.append(baseline_df)
        if os.path.isdir(curr_dir):
            curr_df = read_and_concat(curr_dir, os.listdir(curr_dir), 'curriculum')
            dfs.append(curr_df)
        if os.path.isdir(adapt_curr_dir):
            adapt_curr_df = read_and_concat(adapt_curr_dir, os.listdir(adapt_curr_dir), 'adaptive_curriculum')
            dfs.append(adapt_curr_df)

        # Combine the dataframes
        combined_df = pd.concat(dfs).reset_index(drop=True)

        # Exclude results from task 0
        combined_df = combined_df[combined_df["end_task"] != "0"]

        # Remove underscore for figures
        combined_df["type"] = combined_df["type"].replace("adaptive_curriculum", "adaptive curriculum")
    
    # Create figures folder
    if not os.path.isdir("figures/" + folder):
        os.makedirs("figures/" + folder)

    # Function that creates standard line plots for several metrics
    def create_plot(combined_df, curr_changes=curr_changes, additional_folder = "", additional_file_text = "", additional_title_text = ""):
        for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret']:
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5), dpi=200)
            
            # Include a zoomed in cost curve
            zoomed = ""
            if metric == 'cost_zoom':
                metric = 'cost'
                plt.ylim(0, 2 * cost_limit)
                zoomed = "_zoom"

            # Plot the lines
            sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
            if include_seeds:
                if include_repetitions:
                    ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', errorbar=None, estimator=None, legend=False)
                else:
                    ax = sns.lineplot(data=combined_df.groupby(["step", "Algorithm", "type", "seed"]).mean(), x='step', y=metric, hue='Algorithm', 
                                    style='type', units='seed', errorbar=None, estimator=None, legend=False)

            # Plot the epochs at which a task change occurs
            for change in curr_changes:
                plt.axvline(x=change, color="gray", linestyle='-')

            # Plot the cost limit
            if metric == 'cost':
                plt.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

            # Save the plot
            plt.legend(loc=(1.01, 0.01), ncol=1)
            if include_seeds:
                plt.setp(ax.lines[2:], alpha=0.2)
            plt.tight_layout(pad=2)
            plt.title(f"{metric.replace('_', ' ').capitalize() if metric != 'length' else 'Episode' + metric}s of{' ' + additional_title_text if additional_title_text != '' else ''} agents using curriculum and baseline agent during evaluation")
            plt.xlabel("x1000 Steps")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
                os.makedirs(f"figures/{folder}/{additional_folder}")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}_eval.png")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}_eval.pdf")
            plt.close()

    # Create plots for whole data
    create_plot(combined_df=combined_df)

    # Create plots for each algorithm
    for algo in combined_df["Algorithm"].unique():
        create_plot(combined_df=combined_df[combined_df['Algorithm'] == algo], additional_folder=algo, additional_title_text=algo)

    # Create plots for each environment
    for end_task in combined_df["end_task"].unique():
        if end_task == "T":
            idx = 6
        else:
            idx = int(end_task)
        create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], curr_changes=curr_changes[:idx], additional_folder="HM" + str(end_task), 
                    additional_title_text="HM" + str(end_task))

    # Function that creates a grid using the last three end tasks and the three main metrics
    def create_subplot_grid_3_by_3(combined_df, curr_changes, additional_folder="", additional_file_text=""):
        end_tasks = combined_df['end_task'].unique()
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 6.4), dpi=200)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            for ax, end_task in zip(ax_row, end_tasks[-3:]):
                sns.set_style("whitegrid")
                
                # Plot the line for this end task and metric
                sns.lineplot(data=combined_df[combined_df['end_task'] == end_task], x='step', y=metric, hue='type', errorbar="sd" if use_std else "se", ax=ax)
                
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
                    ax.set_ylabel(metric.replace('_', ' ').capitalize())
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}")
                if metric == "cost" and end_task == "T":
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=(1.01, 0.01), ncol=1)
        
        # Save the plot
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid_eval.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid_eval.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subplot_grid_3_by_3(combined_df=combined_df, curr_changes=curr_changes)
    create_subplot_grid_3_by_3(combined_df=combined_df[combined_df["seed"].isin(map(str, [5905, 7337, 572, 5689, 3968]))], curr_changes=curr_changes, additional_file_text="fair_")

    # Function that creates a grid using the all end tasks and the three main metrics
    def create_subplot_grid_6_by_3(combined_df, curr_changes, additional_folder="", additional_file_text=""):
        end_tasks = combined_df['end_task'].unique()
        
        # Create a 6x3 subplot grid
        fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(12, 12.8), dpi=200)
        for ax_row, end_task in zip(axes, end_tasks):
            for ax, metric in zip(ax_row, ["return", "cost", "regret"]):
                sns.set_style("whitegrid")
                
                # Plot the line for this end task and metric
                sns.lineplot(data=combined_df[combined_df['end_task'] == end_task], x='step', y=metric, hue='type', errorbar="sd" if use_std else "se", ax=ax)
                
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
                if end_task == "1":
                    ax.set_title(metric.replace('_', ' ').capitalize(), fontsize=14)
                if metric == "return":
                    ax.set_ylabel(f"Task {end_task}", rotation=0, loc="top", fontsize=14)
                else:
                    ax.set_ylabel('')
                if metric == "cost" and end_task == "T":
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=(0.01, -1.2), ncol=1)
        
        # Save the plot
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid_eval.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid_eval.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subplot_grid_6_by_3(combined_df=combined_df[combined_df["Algorithm"] == "PPOLag"], curr_changes=curr_changes, additional_folder="PPOLag")

    # Function that creates a grid of spiderplots using the all end tasks and the three main metrics
    def create_subspiderplot_grid(combined_df, curr_changes, additional_folder="", additional_file_text=""):
        end_tasks = combined_df['end_task'].unique()
        theta = radar_factory(4, frame='polygon')
        algorithms = ["FOCOPS", "CUP", "PPOEarlyTerminated", "PPOLag"]
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(21, 10), dpi=200, subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=-0.85, hspace=-0.95)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            for ax, end_task in zip(ax_row, end_tasks):
                # Add PPO
                filtered_df = combined_df[(combined_df["Algorithm"] == "PPO") & (combined_df['end_task'] == end_task)]
                mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                ppo_metrics = [mean_df[metric].iloc[-1] for _ in range(len(algorithms))]
                ax.plot(theta, ppo_metrics, color="red")

                # Plot the main algorithms
                for algorithm_type in combined_df["type"].unique():
                    metrics = []
                    for algorithm in algorithms:
                        filtered_df = combined_df[(combined_df["Algorithm"] == algorithm) & (combined_df['type'] == algorithm_type) & (combined_df['end_task'] == end_task)]
                        mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                        metrics.append(mean_df[metric].iloc[-1])
                    ax.plot(theta, metrics)
                    ax.fill(theta, metrics, alpha=0.15, label='_nolegend_')

                # Set labels with hard-coded spacing
                ax.set_varlabels(["FOCOPS", "CUP   ", "PPOEarlyTerminated", "     PPOLag"], fontsize=13)
                
                # Create titles and labels
                ax.set_xlabel("x1000 Steps")
                if end_task == "1":
                    ax.set_ylabel(metric.replace('_', ' ').capitalize(), loc="top", rotation=0, fontsize=18)
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}", fontsize=18)

        # Create legend
        labels = ("PPO", "baseline", "curriculum", "adaptive\ncurriculum")
        legend = axes[0, 0].legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize=14)
    
        # Save the plot
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid_eval.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid_eval.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subspiderplot_grid(combined_df=combined_df[combined_df["Algorithm"].isin(["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO"])], 
                              curr_changes=curr_changes, additional_file_text="new_")
    
    return combined_df

def print_incremental_results(folder, train_df, eval_df, save_freq, additional_folder = ""):
    # Function to print several metrics into a .txt file
    for (algorithm, algorithm_type), filtered_train_df in train_df.groupby(["Algorithm", 'type']):
        filtered_eval_df = eval_df[(eval_df["Algorithm"] == algorithm) & (eval_df['type'] == algorithm_type)]
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

        if not os.path.isdir(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{algorithm_type}_metrics"):
            os.makedirs(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{algorithm_type}_metrics")
        with open(os.path.join(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}", 
                               f"{algorithm_type}_metrics/{algorithm}-metrics.txt"), 'w') as file:
            file.write("Last epoch results:\n")
            file.write(f"Return: {return_}\n")
            file.write(f"Cost: {cost}\n")
            file.write(f"Evaluation return: {eval_return}\n")
            file.write(f"Evaluation cost: {eval_cost}\n")
            file.write(f"Evaluation episode length: {eval_length}\n")
            file.write("\nAll epochs results:\n")
            file.write(f"AUC of the cost curve: {auc_cost}\n")
            file.write(f"AUC of the evaluation cost curve: {auc_eval_cost}\n")
            file.write(f"Cost regret: {regret_train}\n")
            file.write(f"Evaluation cost regret: {regret_eval}\n")
            file.close()


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    Code taken from matplotlib: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, verticalalignment="top", **kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
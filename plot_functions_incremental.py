
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

def plot_incremental_train(folder, curr_changes, cost_limit, combined_df=None, include_weak=False, include_seeds=False, use_std=False):
    # Get folder names for all algorithms
    baseline_dir = "results/" + folder + "/baseline"
    curr_dir = "results/" + folder + "/curriculum"
    adapt_curr_dir = f"results/" + folder + "/adaptive_curriculum"

    # Function to read progress csv and concatenate
    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            for path in paths:
                path = path.replace("\\", "/")
                # print(path)
                df = pd.read_csv(os.path.join(path, "progress.csv")).rename(columns=
                    {"Metrics/EpRet": "return", "Metrics/EpCost": "cost", "Metrics/EpLen": "length"}
                )[['return', 'cost', 'length']]
                df['Algorithm'] = algorithm.split("-")[0]
                end_version_pattern = r'HMR?(\d+|T)'
                end_version = re.search(end_version_pattern, algorithm.split("-")[1])
                df['end_task'] = end_version.group(1)
                df['type'] = algorithm_type
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-1].split("-")[1]
                df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)
                df = df.sort_index()
                df['regret'] = df['regret_per_epoch'].cumsum()
                dfs.append(df)
        return pd.concat(dfs)

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

        # Combine both baseline and curriculum dataframes
        combined_df = pd.concat(dfs).reset_index(names="step")
    
    if not os.path.isdir("figures/" + folder):
        os.makedirs("figures/" + folder)
        
    last_change = curr_changes[-1]

    def create_plot(combined_df, curr_changes=curr_changes, additional_folder = "", additional_file_text = "", additional_title_text = ""):
        for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret']:
            # Plotting using Seaborn
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5), dpi=200)

            # include a zoomed in cost curve
            zoomed = ""
            if metric == 'cost_zoom':
                metric = 'cost'
                plt.ylim(0, 2 * cost_limit)
                zoomed = "_zoom"

            sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
            if include_seeds:
                ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', estimator=None, legend=False)

            for change in curr_changes:
                plt.axvline(x=change, color="gray", linestyle='-')

            if metric == 'cost':
                plt.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

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

    # # Create plots for whole data
    # create_plot(combined_df=combined_df)

    # # Create plots for each algorithm
    # for algo in combined_df["Algorithm"].unique():
    #     create_plot(combined_df=combined_df[combined_df['Algorithm'] == algo], additional_folder=algo, additional_title_text=algo)

    # print(combined_df["end_task"].unique())
    # for end_task in combined_df["end_task"].unique():
    #     if end_task == "T":
    #         idx = 6
    #     else:
    #         idx = int(end_task)
        # create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], curr_changes=curr_changes[:idx], additional_folder="HM" + str(end_task), 
        #             additional_title_text="HM" + str(end_task))
        
        # create_plot(combined_df=combined_df[(combined_df['end_task'] == end_task) & (combined_df['seed'].isin(map(str,[5905, 7337, 572, 5689, 3968])))], curr_changes=curr_changes[:idx], 
        #             additional_folder="FairHM" + str(end_task), additional_title_text="FairHM" + str(end_task))
        
    def create_subplot_grid(combined_df, curr_changes, additional_folder="", additional_file_text="", additional_title_text=""):
        end_tasks = combined_df['end_task'].unique()
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(24, 6.4), dpi=200)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            for ax, end_task in zip(ax_row, end_tasks):
                sns.set_style("whitegrid")
                
                sns.lineplot(data=combined_df[combined_df['end_task'] == end_task], x='step', y=metric, hue='type', errorbar="sd" if use_std else "se", ax=ax)
                
                if end_task == "T":
                    idx = 6
                else:
                    idx = int(end_task)
                for change in curr_changes[:idx]:
                    ax.axvline(x=change, color="gray", linestyle='-')
                
                if metric == 'cost':
                    ax.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')
                    
                ax.set_xlabel("x1000 Steps")
                ax.get_legend().remove()
                if end_task == "1":
                    ax.set_ylabel(metric.replace('_', ' ').capitalize())
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}")
                if metric == "cost" and end_task == "T":
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=(1.01, 0.01), ncol=1)
        
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subplot_grid(combined_df=combined_df[combined_df["Algorithm"] == "PPOLag"], curr_changes=curr_changes, additional_folder="PPOLag")

    def create_subspiderplot_grid(combined_df, curr_changes, additional_folder="", additional_file_text="", additional_title_text=""):
        end_tasks = combined_df['end_task'].unique()
        N = 4
        theta = radar_factory(N, frame='polygon')
        algorithms = ["FOCOPS", "CUP", "PPOEarlyTerminated", "PPOLag"]
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(20, 10), dpi=200, subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=-0.3, hspace=-0.5)
        print(axes)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            print(ax_row)
            print(end_tasks)
            for ax, end_task in zip(ax_row, end_tasks):
                print(ax)

                # Add PPO
                filtered_df = combined_df[(combined_df["Algorithm"] == "PPO") & (combined_df['end_task'] == end_task)]
                mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                ppo_metrics = [mean_df[metric].iloc[-1] for _ in range(len(algorithms))]
                ax.plot(theta, ppo_metrics, color="black")

                max_metric = 0
                for algorithm_type in combined_df["type"].unique():
                    metrics = []
                    for algorithm in algorithms:
                        filtered_df = combined_df[(combined_df["Algorithm"] == algorithm) & (combined_df['type'] == algorithm_type) & (combined_df['end_task'] == end_task)]
                        mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                        metrics.append(mean_df[metric].iloc[-1])
                    print(metrics)
                    max_metric = max(max_metric, max(metrics))
                    ax.plot(theta, metrics)
                    ax.fill(theta, metrics, alpha=0.25, label='_nolegend_')

                # TODO Double-check that this is tied to the correct labels:
                ax.set_varlabels(algorithms)
                # if max_metric < 20:
                #     max_metric = round(max_metric * 2 + 1, 0) / 2
                # elif max_metric < 100:
                #     max_metric = round(max_metric + 1, -1)
                # elif max_metric < 1000:
                #     max_metric = round(max_metric + 1, -1)
                # else:
                #     max_metric = round(max_metric, -3)
                # ax.set_rgrids([i * max_metric for i in [0.2, 0.4, 0.6, 0.8, 1.0]])
                
                if end_task == "T":
                    idx = 6
                else:
                    idx = int(end_task)
                # for change in curr_changes[:idx]:
                #     ax.axvline(x=change, color="gray", linestyle='-')
                
                # if metric == 'cost':
                #     ax.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')
                    
                ax.set_xlabel("x1000 Steps")
                # ax.get_legend().remove()
                if end_task == "1":
                    ax.set_ylabel(metric.replace('_', ' ').capitalize(), loc="top")
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}")
                # if metric == "cost" and end_task == "T":
                #     handles, labels = ax.get_legend_handles_labels()
                #     ax.legend(handles, labels, loc=(1.01, 0.01), ncol=1)

        labels = ("PPO", "baseline", "curriculum")
        legend = axes[0, 0].legend(labels, loc=(0.9, .95),
                                labelspacing=0.1)
    
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid.pdf")
        plt.close()

    # Call the function to create the grid of plots
    # create_subspiderplot_grid(combined_df=combined_df[combined_df["Algorithm"].isin(["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO"])], curr_changes=curr_changes)

    return combined_df

def plot_incremental_eval(folder, curr_changes, cost_limit, combined_df=None, include_weak=False, include_seeds=False, include_repetitions=False, use_std=False):
    def extract_values(pattern, text):
        return [float(match.group(1)) for match in re.finditer(pattern, text)]

    baseline_dir = "results/" + folder + "/baseline"
    curr_dir = "results/" + folder + "/curriculum"
    adapt_curr_dir = f"results/" + folder + "/adaptive_curriculum"

    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            seed_paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            eval_paths = [os.path.join(path, "evaluation") for path in seed_paths]

            for path in eval_paths:
                path = path.replace("\\", "/")
                # print(path)
                epochs = [entry.name for entry in os.scandir(path)]

                returns = []
                costs = []
                lengths = []
                steps = []

                reps = 0

                for epoch in epochs:
                    with open(os.path.join(path, epoch, "result.txt"), 'r') as file:
                        data = file.read()

                        return_ = extract_values(r'Episode reward: ([\d\.-]+)', data)
                        cost_ = extract_values(r'Episode cost: ([\d\.-]+)', data)
                        length_ = extract_values(r'Episode length: ([\d\.-]+)', data)

                        returns += return_
                        costs += cost_
                        lengths += length_

                        if reps == 0:
                            reps = len(return_)

                        index = int(epoch.split("-")[1])
                        steps += [index for i in range(reps)]

                df = pd.DataFrame({'return': returns, 'cost': costs, 'length': lengths, 'step': steps})
                df['Algorithm'] = algorithm.split("-")[0]
                end_version_pattern = r'HMR?A?(\d+|T)'
                end_version = re.search(end_version_pattern, algorithm.split("-")[1])
                df['end_task'] = end_version.group(1)
                df['type'] = algorithm_type
                # print("train: ", str(path).split("/" if "/" in str(path) else '\\')[-1].split("-")[1])
                # print("eval: ", str(path).split("/" if "/" in str(path) else '\\')[-2].split("-")[1])
                df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-2].split("-")[1]
                df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)

                df['step'] = pd.to_numeric(df['step'])
                df = df.sort_values(by=['step']).reset_index()
                avg_regret_per_epoch= df.groupby(df.index // reps)['regret_per_epoch'].mean()
                df['regret'] = avg_regret_per_epoch.cumsum().repeat(reps).reset_index(drop=True)
                dfs.append(df)
        return pd.concat(dfs)

    if combined_df is not None:
        # Combine both baseline and curriculum dataframes
        # combined_df = pd.concat([combined_df, pd.concat([baseline_df, curr_df]).reset_index(drop=True)])
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

        combined_df = pd.concat(dfs).reset_index(drop=True)
    
    if not os.path.isdir("figures/" + folder):
        os.makedirs("figures/" + folder)

    def create_plot(combined_df, curr_changes=curr_changes, additional_folder = "", additional_file_text = "", additional_title_text = ""):
        for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret']:
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5), dpi=200)
            
            # include a zoomed in cost curve
            zoomed = ""
            if metric == 'cost_zoom':
                metric = 'cost'
                plt.ylim(0, 2 * cost_limit)
                zoomed = "_zoom"

            sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', errorbar="sd" if use_std else "se")
            if include_seeds:
                if include_repetitions:
                    ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='Algorithm', style='type', units='seed', errorbar=None, estimator=None, legend=False)
                else:
                    ax = sns.lineplot(data=combined_df.groupby(["step", "Algorithm", "type", "seed"]).mean(), x='step', y=metric, hue='Algorithm', 
                                    style='type', units='seed', errorbar=None, estimator=None, legend=False)

            for change in curr_changes:
                plt.axvline(x=change, color="gray", linestyle='-')

            if metric == 'cost':
                plt.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

            plt.legend(loc=(1.01, 0.01), ncol=1)
            if include_seeds:
                plt.setp(ax.lines[2:], alpha=0.2)
            plt.tight_layout(pad=2)
            plt.title(f"{metric.replace('_', ' ').capitalize() if metric != 'length' else 'Episode' + metric}s of{' ' + additional_title_text if additional_title_text != '' else ''} agents using curriculum and baseline agent during evalutaion")
            plt.xlabel("x1000 Steps")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
                os.makedirs(f"figures/{folder}/{additional_folder}")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}_eval.png")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}_eval.pdf")
            plt.close()

    # # Create plots for whole data
    # create_plot(combined_df=combined_df)

    # Create plots for each environment
    # print(combined_df["end_task"].unique())
    # for end_task in combined_df["end_task"].unique():
    #     if end_task == "T":
    #         idx = 6
    #     else:
    #         idx = int(end_task)
    #     create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], curr_changes=curr_changes[:idx], additional_folder="HM" + str(end_task), 
    #                 additional_title_text="HM" + str(end_task))
        
    #     create_plot(combined_df=combined_df[(combined_df['end_task'] == end_task) & (combined_df['seed'].isin([5905, 7337, 572, 5689, 3968]))], curr_changes=curr_changes[:idx], 
    #                 additional_folder="FairHM" + str(end_task), additional_title_text="FairHM" + str(end_task))

    # # Create plots for each algorithm
    # for algo in combined_df["Algorithm"].unique():
    #     create_plot(combined_df=combined_df[combined_df['Algorithm'] == algo], additional_folder=algo, additional_title_text=algo)

    def create_subplot_grid(combined_df, curr_changes, additional_folder="", additional_file_text="", additional_title_text=""):
        end_tasks = combined_df['end_task'].unique()
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(24, 6.4), dpi=200)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            for ax, end_task in zip(ax_row, end_tasks):
                sns.set_style("whitegrid")
                
                sns.lineplot(data=combined_df[combined_df['end_task'] == end_task], x='step', y=metric, hue='type', errorbar="sd" if use_std else "se", ax=ax)
                
                if end_task == "T":
                    idx = 6
                else:
                    idx = int(end_task)
                for change in curr_changes[:idx]:
                    ax.axvline(x=change, color="gray", linestyle='-')
                
                if metric == 'cost':
                    ax.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')
                    
                ax.set_xlabel("x1000 Steps")
                ax.get_legend().remove()
                if end_task == "1":
                    ax.set_ylabel(metric.replace('_', ' ').capitalize())
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}")
                if metric == "cost" and end_task == "T":
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=(1.01, 0.01), ncol=1)
        
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid_eval.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}grid_eval.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subplot_grid(combined_df=combined_df[combined_df["Algorithm"] == "PPOLag"], curr_changes=curr_changes, additional_folder="PPOLag")

    def create_subspiderplot_grid(combined_df, curr_changes, additional_folder="", additional_file_text="", additional_title_text=""):
        end_tasks = combined_df['end_task'].unique()
        N = 4
        theta = radar_factory(N, frame='polygon')
        algorithms = ["FOCOPS", "CUP", "PPOEarlyTerminated", "PPOLag"]
        
        # Create a 3x3 subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(20, 10), dpi=200, subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=-0.3, hspace=-0.5)
        print(axes)
        for ax_row, metric in zip(axes, ["return", "cost", "regret"]):
            print(ax_row)
            print(end_tasks)
            for ax, end_task in zip(ax_row, end_tasks):
                print(ax)

                # Add PPO
                filtered_df = combined_df[(combined_df["Algorithm"] == "PPO") & (combined_df['end_task'] == end_task)]
                mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                ppo_metrics = [mean_df[metric].iloc[-1] for _ in range(len(algorithms))]
                ax.plot(theta, ppo_metrics, color="black")

                max_metric = 0
                for algorithm_type in combined_df["type"].unique():
                    metrics = []
                    for algorithm in algorithms:
                        filtered_df = combined_df[(combined_df["Algorithm"] == algorithm) & (combined_df['type'] == algorithm_type) & (combined_df['end_task'] == end_task)]
                        mean_df = filtered_df.groupby(["step"]).mean(numeric_only=True)
                        metrics.append(mean_df[metric].iloc[-1])
                    print(metrics)
                    max_metric = max(max_metric, max(metrics))
                    ax.plot(theta, metrics)
                    ax.fill(theta, metrics, alpha=0.25, label='_nolegend_')

                # TODO Double-check that this is tied to the correct labels:
                ax.set_varlabels(algorithms)
                # if max_metric < 20:
                #     max_metric = round(max_metric * 2 + 1, 0) / 2
                # elif max_metric < 100:
                #     max_metric = round(max_metric + 1, -1)
                # elif max_metric < 1000:
                #     max_metric = round(max_metric + 1, -1)
                # else:
                #     max_metric = round(max_metric, -3)
                # ax.set_rgrids([i * max_metric for i in [0.2, 0.4, 0.6, 0.8, 1.0]])
                
                if end_task == "T":
                    idx = 6
                else:
                    idx = int(end_task)
                # for change in curr_changes[:idx]:
                #     ax.axvline(x=change, color="gray", linestyle='-')
                
                # if metric == 'cost':
                #     ax.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')
                    
                ax.set_xlabel("x1000 Steps")
                # ax.get_legend().remove()
                if end_task == "1":
                    ax.set_ylabel(metric.replace('_', ' ').capitalize(), loc="top")
                else:
                    ax.set_ylabel('')
                if metric == "return":
                    ax.set_title(f"Task {end_task}")
                # if metric == "cost" and end_task == "T":
                #     handles, labels = ax.get_legend_handles_labels()
                #     ax.legend(handles, labels, loc=(1.01, 0.01), ncol=1)

        labels = ("PPO", "baseline", "curriculum")
        legend = axes[0, 0].legend(labels, loc=(0.9, .95),
                                labelspacing=0.1)
    
        plt.tight_layout(pad=2)
        if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
            os.makedirs(f"figures/{folder}/{additional_folder}")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid_eval.png")
        plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}spider_grid_eval.pdf")
        plt.close()

    # Call the function to create the grid of plots
    create_subspiderplot_grid(combined_df=combined_df[combined_df["Algorithm"].isin(["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO"])], curr_changes=curr_changes)

    return combined_df

def print_incremental_eval(folder, train_df, eval_df, save_freq, cost_limit, additional_folder = ""):
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
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

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
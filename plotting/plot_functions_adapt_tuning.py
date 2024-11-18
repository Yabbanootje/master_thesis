
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline, BSpline

def plot_adapt_tune_train(folder, cost_limit, include_seeds=False, use_std=False):
    # Get folder names for all algorithms
    adapt_curr_dir = f"results/" + folder + "/adaptive_curriculum"
    
    # Function to read progress csv and concatenate adaptive results into a dataframe
    def read_and_concat_adaptive(directory):
        dfs = []
        for beta_dir in os.listdir(directory):
            beta_path = os.path.join(directory, beta_dir)
            beta_value = beta_dir.split("-")[1]
            for kappa_dir in os.listdir(beta_path):
                kappa_path = os.path.join(beta_path, kappa_dir)
                kappa_value = kappa_dir.split("-")[1]
                algorithms = os.listdir(kappa_path)
                for algorithm in algorithms:
                    paths = [entry.path for entry in os.scandir(os.path.join(kappa_path, algorithm))]
                    for path in paths:
                        # For each repetition of the parameter combination
                        path = path.replace("\\", "/")
                        df = pd.read_csv(os.path.join(path, "progress.csv")).rename(columns=
                            {"Metrics/EpRet": "return", "Metrics/EpCost": "cost", "Metrics/EpLen": "length", "Current_task": "current_task"}
                        )[['return', 'cost', 'length', 'current_task']]
                        df['Algorithm'] = algorithm.split("-")[0]
                        end_version_pattern = r'HMR?A?(\d+|T)'
                        end_version = re.search(end_version_pattern, algorithm.split("-")[1])
                        df['end_task'] = str(end_version.group(1))
                        df['type'] = 'adaptive_curriculum'
                        df['beta'] = beta_value
                        df['kappa'] = kappa_value
                        df['beta_kappa'] = str(beta_value) + "-" + str(kappa_value)
                        df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-1].split("-")[1]
                        df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)
                        df = df.sort_index()
                        df['regret'] = df['regret_per_epoch'].cumsum()
                        dfs.append(df)
        return pd.concat(dfs)

    # Put results in dataframe
    combined_df = read_and_concat_adaptive(adapt_curr_dir).reset_index(names="step")
    
    # Create figures folder
    if not os.path.isdir(f"figures/" + folder):
        os.makedirs(f"figures/" + folder)

    # Function that creates standard line plots for several metrics
    def create_plot(combined_df, smooth = False, additional_folder = "", additional_file_text = "", additional_title_text = "", include_seeds=False):
        for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret', 'current_task']:
            # Plotting using Seaborn
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5), dpi=200)

            # Include a zoomed in cost curve
            zoomed = ""
            if metric == 'cost_zoom':
                metric = 'cost'
                plt.ylim(0, 2 * cost_limit)
                zoomed = "_zoom"

            if smooth:
                # Smooth out results and plot the lines
                for hue in combined_df["beta_kappa"].unique():
                    sns.regplot(data=combined_df[combined_df["beta_kappa"] == hue], x='step', y=metric, scatter=False, label=hue, order=20)
            else:
                # Plot the lines
                sns.lineplot(data=combined_df, x='step', y=metric, hue='beta_kappa', errorbar="sd" if use_std else "se")
            if include_seeds:
                ax = sns.lineplot(data=combined_df, x='step', y=metric, hue='seed', estimator=None, legend=False)

            # Plot the cost limit
            if metric == 'cost':
                plt.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

            # Save the plot
            plt.legend(loc=(1.01, 0.01), ncol=1)
            if include_seeds:
                plt.setp(ax.lines[2:], alpha=0.2)
            plt.tight_layout(pad=2)
            plt.title(f"{metric.replace('_', ' ').capitalize()}s of{' ' + additional_title_text if additional_title_text != '' else ''} agents using an adaptive curriculum")
            plt.xlabel("x1000 Steps")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
                os.makedirs(f"figures/{folder}/{additional_folder}")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}.png")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}.pdf")
            plt.close()

    # Create plots for whole data
    create_plot(combined_df=combined_df)

    # Create plots for each environment
    for end_task in combined_df["end_task"].unique():
        create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], smooth=True, additional_folder="HM" + end_task, 
                    additional_title_text="HM" + end_task, additional_file_text="smooth_")

    # Create plots for each parameter combination
    for beta_kappa in combined_df["beta_kappa"].unique():
        create_plot(combined_df=combined_df[combined_df['beta_kappa'] == beta_kappa], additional_folder=beta_kappa, additional_title_text=beta_kappa, include_seeds=True)

    # Create plots only containing the results of the five best parameter combinations
    create_plot(combined_df=combined_df[(combined_df['end_task'] == "T") & (combined_df['beta_kappa'].isin(["1.5-5", "1.5-20", "0.5-20", "0.5-10", "1.0-20"]))], 
                smooth=True, additional_folder="best_params", additional_title_text="", additional_file_text="smooth_")

    return combined_df

def plot_adapt_tune_eval(folder, cost_limit, use_std=False):
    # Get folder names for all algorithms
    baseline_dir = f"results/" + folder + "/baseline"
    curr_dir = f"results/" + folder + "/curriculum"
    adapt_curr_dir = f"results/" + folder + "/adaptive_curriculum"

    # Function that extracts values from the logs
    def extract_values(pattern, text):
        return [float(match.group(1)) for match in re.finditer(pattern, text)]

    # Function to read progress csv and concatenate adaptive results into a dataframe
    def read_and_concat_adaptive(directory):
        dfs = []
        for beta_dir in os.listdir(directory):
            beta_path = os.path.join(directory, beta_dir)
            beta_value = beta_dir.split("-")[1]
            for kappa_dir in os.listdir(beta_path):
                kappa_path = os.path.join(beta_path, kappa_dir)
                kappa_value = kappa_dir.split("-")[1]
                algorithms = os.listdir(kappa_path)
                for algorithm in algorithms:
                    seed_paths = [entry.path for entry in os.scandir(os.path.join(kappa_path, algorithm))]
                    eval_paths = [os.path.join(path, "evaluation") for path in seed_paths]

                    for path in eval_paths:
                        # For each repetition of the parameter combination
                        path = path.replace("\\", "/")
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
                        df['type'] = 'adaptive_curriculum'
                        df['beta'] = beta_value
                        df['kappa'] = kappa_value
                        df['beta_kappa'] = str(beta_value) + "-" + str(kappa_value)
                        df['seed'] = str(path).split("/" if "/" in str(path) else '\\')[-2].split("-")[1]
                        df['regret_per_epoch'] = (df["cost"] - cost_limit).clip(lower=0.0)
                        df['step'] = pd.to_numeric(df['step'])
                        df = df.sort_values(by=['step']).reset_index()
                        avg_regret_per_epoch= df.groupby(df.index // reps)['regret_per_epoch'].mean()
                        df['regret'] = avg_regret_per_epoch.cumsum().repeat(reps).reset_index(drop=True)
                        dfs.append(df)
        return pd.concat(dfs)

    # Put results in dataframe
    combined_df = read_and_concat_adaptive(adapt_curr_dir).reset_index(drop=True)
    
    # Create figures folder
    if not os.path.isdir(f"figures/" + folder):
        os.makedirs(f"figures/" + folder)

    # Function that creates standard line plots for several metrics
    def create_plot(combined_df, smooth = False, additional_folder = "", additional_file_text = "", additional_title_text = ""):
        for metric in ['return', 'cost', 'length', 'cost_zoom', 'regret']:
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5), dpi=200)
            
            # Include a zoomed in cost curve
            zoomed = ""
            if metric == 'cost_zoom':
                metric = 'cost'
                plt.ylim(0, 2 * cost_limit)
                zoomed = "_zoom"

            if smooth:
                # Smooth out results and plot the lines
                for hue in combined_df["beta_kappa"].unique():
                    sns.regplot(data=combined_df[combined_df["beta_kappa"] == hue], x='step', y=metric, scatter=False, label=hue, order=20)
            else:
                # Plot the lines
                sns.lineplot(data=combined_df, x='step', y=metric, hue='beta_kappa', errorbar="sd" if use_std else "se")

            # Plot the cost limit
            if metric == 'cost':
                plt.axhline(y=cost_limit, color='black', linestyle=':', label=f'Cost Limit ({cost_limit})')

            # Save the plot
            plt.legend(loc=(1.01, 0.01), ncol=1)
            plt.tight_layout(pad=2)
            plt.title(f"{metric.replace('_', ' ').capitalize() if metric != 'length' else 'Episode' + metric}s of{' ' + additional_title_text if additional_title_text != '' else ''} agents using an adaptive curriculum during evaluation")
            plt.xlabel("x1000 Steps")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            if not os.path.isdir(f"figures/{folder}/{additional_folder}"):
                os.makedirs(f"figures/{folder}/{additional_folder}")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}_eval.png")
            plt.savefig(f"figures/{folder}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}{metric}s{zoomed}_eval.pdf")
            plt.close()

    # Create plots for whole data
    create_plot(combined_df=combined_df)

    # Create plots for each environment
    for end_task in combined_df["end_task"].unique():
        create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], smooth = True, additional_folder="HM" + end_task, additional_title_text="HM" + end_task, additional_file_text="smooth_")

    # Create plots for each parameter combination
    for beta_kappa in combined_df["beta_kappa"].unique():
        create_plot(combined_df=combined_df[combined_df['beta_kappa'] == beta_kappa], additional_folder=beta_kappa, additional_title_text=beta_kappa, include_seeds=True) 

    # Create plots only containing the results of the five best parameter combinations
    create_plot(combined_df=combined_df[(combined_df['end_task'] == "T") & (combined_df['beta_kappa'].isin(["1.5-5", "1.5-20", "0.5-20", "0.5-10", "1.0-20"]))], 
                smooth=True, additional_folder="best_params", additional_title_text="", additional_file_text="smooth_")

    return combined_df

def print_adapt_tune_results(folder, train_df, eval_df, save_freq):
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

        if not os.path.isdir(f"figures/{folder}/{algorithm_type}_metrics"):
            os.makedirs(f"figures/{folder}/{algorithm_type}_metrics")
        with open(os.path.join(f"figures/{folder}/", f"{algorithm_type}_metrics/{algorithm}-metrics.txt"), 'w') as file:
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
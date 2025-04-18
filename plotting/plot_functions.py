
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_train(folder, curr_changes, cost_limit, combined_df=None, include_seeds=False, use_std=False):
    # Get folder names for all algorithms
    baseline_dir = f"results/" + folder + "/baseline"
    curr_dir = f"results/" + folder + "/curriculum"
    adapt_curr_dir = f"results/" + folder + "/adaptive_curriculum"

    # Function to read progress csv and concatenate results into a dataframe
    def read_and_concat(directory, algorithms, algorithm_type):
        dfs = []
        for algorithm in algorithms:
            paths = [entry.path for entry in os.scandir(os.path.join(directory, algorithm))]
            for path in paths:
                # For each repetition of the algorithm
                path = path.replace("\\", "/")
                df = pd.read_csv(os.path.join(path, "progress.csv")).rename(columns=
                    {"Metrics/EpRet": "return", "Metrics/EpCost": "cost", "Metrics/EpLen": "length"}
                )[['return', 'cost', 'length']]
                df['Algorithm'] = algorithm.split("-")[0]
                end_version_pattern = r'HMR?A?(\d+|T)'
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
    
    # Create figures folder
    if not os.path.isdir(f"figures/" + folder):
        os.makedirs(f"figures/" + folder)

    # Function that creates standard line plots for several metrics
    def create_plot(combined_df, additional_folder = "", additional_file_text = "", additional_title_text = ""):
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

    # # Create plots for each environment
    for end_task in combined_df["end_task"].unique():
        create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], additional_folder="HM" + end_task, additional_title_text="HM" + end_task)

    # Create plots for each algorithm
    for algo in combined_df["Algorithm"].unique():
        create_plot(combined_df=combined_df[combined_df['Algorithm'] == algo], additional_folder=algo, additional_title_text=algo)

    return combined_df

def plot_eval(folder, curr_changes, cost_limit, combined_df=None, include_seeds=False, include_repetitions=False, use_std=False):
    # Get folder names for all algorithms
    baseline_dir = f"results/" + folder + "/baseline"
    curr_dir = f"results/" + folder + "/curriculum"
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
    
    # Create figures folder
    if not os.path.isdir(f"figures/" + folder):
        os.makedirs(f"figures/" + folder)

    # Function that creates standard line plots for several metrics
    def create_plot(combined_df, additional_folder = "", additional_file_text = "", additional_title_text = ""):
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

    # Create plots for each environment
    for end_task in combined_df["end_task"].unique():
        create_plot(combined_df=combined_df[combined_df['end_task'] == end_task], additional_folder="HM" + end_task, additional_title_text="HM" + end_task)

    # Create plots for each algorithm
    for algo in combined_df["Algorithm"].unique():
        create_plot(combined_df=combined_df[combined_df['Algorithm'] == algo], additional_folder=algo, additional_title_text=algo)

    return combined_df

def print_results(folder, train_df, eval_df, save_freq):
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
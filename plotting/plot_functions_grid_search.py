import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(folder_base, algo_type, parameters, promising_parameters, promising_parameters_curriculum):
    # Load results
    last_means = pd.read_csv(f"./figures/{folder_base}/last_means.csv").set_index(parameters)

    # Sort by evaluation return
    if algo_type == "baseline":
        last_means = last_means.sort_values(by=["Evaluation Return"])[["Return", "Cost", "Regret", "Evaluation Return", 
                                                                       "Evaluation Cost", "Evaluation Regret"]]
    else:
        last_means = last_means.sort_values(by=["Evaluation Return Curr"])[["Return Curr", "Cost Curr", "Regret Curr", 
                                                                            "Evaluation Return Curr", "Evaluation Cost Curr", 
                                                                            "Evaluation Regret Curr"]]

    # Get annotation for heatmap
    annotation = last_means.to_numpy()

    # Normalize columns
    for column in last_means.columns:
        if "Cost" in column:
            last_means[column] = np.log(last_means[column] + 1)
        if "Cost" in column or "Regret" in column:
            last_means[column] = -last_means[column]
        last_means[column] = (last_means[column] - last_means[column].min()) / (last_means[column].max() - last_means[column].min())

    # Plotting the heatmap
    fig = plt.figure(figsize=(12, 13))
    ax_img = plt.imshow(last_means.values, cmap="viridis", aspect="auto")
    ax = ax_img.axes
    plt.grid(False)

    # Add labels and ticks
    plt.title("Heatmap of final epoch performance")
    plt.ylabel("Parameter Combinations\n(lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size)")
    plt.xlabel("Metrics")
    plt.yticks(ticks=np.arange(len(last_means.index)), labels=last_means.index, rotation="horizontal")
    plt.xticks(ticks=np.arange(len(last_means.columns)), labels=["Return", "Cost", "Regret", "Evaluation Return", 
                                                                 "Evaluation Cost", "Evaluation Regret"])

    # Show colorbar
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label="Normalized mean of the performance in the final epoch", size=12)
        
    # Get indices in heatmap corresponding to the promising parameters for curriculum agents
    promising_indices = []
    for index, i in zip(last_means.index, range(len(annotation))):
        if tuple(index[i] for i in range(len(index))) in promising_parameters:
            promising_indices.append(i)

    # Get indices in heatmap corresponding to the promising parameters for baseline agents
    promising_indices_curriculum = []
    for index, i in zip(last_means.index, range(len(annotation))):
        if tuple(index[i] for i in range(len(index))) in promising_parameters_curriculum:
            promising_indices_curriculum.append(i)

    # Put textual values inside of the heatmap
    for i in range(len(annotation)):
        for j in range(len(annotation[0])):
            plt.text(j, i, f"{annotation[i, j]:.2f}", fontsize=14, ha="center", va="center")

    # Color the y-axis labels according to which promising parameters it belongs
    for i in range(len(annotation)):
        if i in promising_indices and i in promising_indices_curriculum:
            ax.get_yticklabels()[i].set_color("red")
        elif i in promising_indices:
            ax.get_yticklabels()[i].set_color("orange")
        elif i in promising_indices_curriculum:
            ax.get_yticklabels()[i].set_color("black")
        else:
            ax.get_yticklabels()[i].set_color("grey")

    plt.tight_layout()
    plt.savefig(f"figures/{folder_base}/{algo_type}_heatmap_log_costs_color_ticks.png")
    plt.savefig(f"figures/{folder_base}/{algo_type}_heatmap_log_costs_color_ticks.pdf")
    plt.close()

# Plot the sorted heatmaps
def plot_sorted_heatmap(folder_base, parameters, metrics, filename_prefix):
    fig, axs = plt.subplots(3, 1, figsize=(13, 7))

    # Load results
    last_means = pd.read_csv(f"./figures/{folder_base}/last_means.csv").set_index(parameters)
    
    for ax, metric in zip(axs, metrics):
        # Sort each metric from best to worst
        ascending = False
        if "Cost" in metric or "Regret" in metric:
            # For cost and regret the lower values are the best
            ascending = True
        metric_base = last_means[metric].sort_values(ascending=ascending).reset_index(drop=True)
        metric_curr = last_means[metric + " Curr"].sort_values(ascending=ascending).reset_index(drop=True)

        # Get annotation for heatmap
        metric_last_means = pd.concat([metric_base, metric_curr], axis=1)
        annotation = metric_last_means.T.to_numpy()

        # Normalize column
        if "Cost" in metric:
            metric_last_means = metric_last_means.applymap(lambda x: np.log(x + 1))
        if "Cost" in metric or "Regret" in metric:
            metric_last_means = metric_last_means.applymap(lambda x: -x)

        # Plotting the heatmap
        im = ax.imshow(metric_last_means.T.values, cmap="viridis", aspect="auto")
        ax.grid(False)

        # Add labels and ticks
        ax.set_title(f"{metric}")
        ax.set_ylabel("Agent type")
        ax.set_xlabel("")
        ax.set_yticks(np.arange(2))
        ax.set_yticklabels(["Baseline", "Curriculum"], rotation="horizontal")

        # Put textual values inside of the heatmap
        for i in range(len(annotation)):
            for j in range(len(annotation[0])):
                ax.text(j, i, f"{annotation[i, j]:.2f}", ha="center", va="center", rotation="vertical")

    # Adjust layout and colorbar
    plt.suptitle("Comparison heatmap of final epoch performance")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison.png")
    plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison.pdf")
    plt.close()


# Plot the bar charts
def plot_bar_chart(folder_base, parameters, metrics, filename_prefix):
    fig, axs = plt.subplots(3, 1, figsize=(13, 7))
    last_means = pd.read_csv(f"./figures/{folder_base}/last_means.csv").set_index(parameters)
    last_means = last_means.sort_values(by="Evaluation Return", ascending=False)
    
    for ax, metric in zip(axs, metrics):
        # Get annotation for heatmap
        print(last_means[[metric, metric + " Curr"]].reset_index(drop=True).head(30))
        metric_last_means = last_means[metric + " Curr"] - last_means[metric]
        metric_last_means = metric_last_means.reset_index(drop=True)
        print(metric_last_means.head(30))

        # Plot the baseline line
        ax.axhline(y=0, color="black", linestyle="-", label="Baseline")

        # Plot the deviation bars
        ax.bar(metric_last_means.index, metric_last_means, label="Deviation", color="skyblue")

        # Add labels and legend
        ax.set_xlabel("Parameter Combination")
        ax.set_ylabel(r"$\Delta$ " + metric)
        ax.set_title(metric)
        ax.set_xlim(-1, 72)
        ax.set_xticks([])

    # Adjust layout and colorbar
    plt.suptitle("Relative final epoch performance of curriculum agents")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison_bar.png")
    plt.savefig(f"./figures/{folder_base}/comparison/{filename_prefix}_comparison_bar.pdf")
    plt.close()
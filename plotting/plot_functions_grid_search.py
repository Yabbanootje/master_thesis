import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(folder_base, algo_type, parameters, promising_parameters, promising_parameters_curriculum):
    # Load results
    last_means = pd.read_csv(f"./figures/{folder_base}/last_means.csv").set_index(parameters)

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
    plt.title("Heatmap of final epoch performance", size=14)
    plt.ylabel("Parameter Combinations\n(cost_limit, lag_multiplier_init, lag_multiplier_lr, update_iters, nn_size)", size=12)
    plt.xlabel("Metrics", size=12)
    plt.yticks(ticks=np.arange(len(last_means.index)), labels=last_means.index, rotation="horizontal", fontsize=11)
    plt.xticks(ticks=np.arange(len(last_means.columns)), labels=[col.replace(" Curr", "") for col in last_means.columns])

    # Show colorbar
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label="Normalized mean of the performance in the final epoch", size=12)
        
    # Get indices in heatmap corresponding to the promising parameters for curriculum agents
    promising_indices_curriculum = []
    for index, i in zip(last_means.index, range(len(annotation))):
        if tuple(index[i] for i in range(len(index)) if i != 0) in promising_parameters_curriculum:
            promising_indices_curriculum.append(i)

    # Get indices in heatmap corresponding to the promising parameters for baseline agents
    promising_indices = []
    for index, i in zip(last_means.index, range(len(annotation))):
        if tuple(index[i] for i in range(len(index)) if i != 0) in promising_parameters:
            promising_indices.append(i)

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
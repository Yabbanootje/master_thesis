import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    eval_episodes = 5
    render_episodes = 3
    cost_limit = 5.0
    steps_per_epoch = 1000
    save_freq = 10
    epochs = 3000
    repetitions = 10
    # baseline_algorithms = []#["PPO", "CPO", "OnCRPO", "CUP", "FOCOPS", "PCPO", "PPOEarlyTerminated", "PPOLag"]
    # curr_algorithms = ["PPOLag"]#["OnCRPO", "CUP", "FOCOPS", "PCPO", "PPOEarlyTerminated", "PPOLag"]
    baseline_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated", "PPO", "CPO"]
    curr_algorithms = ["PPOLag", "FOCOPS", "CUP", "PPOEarlyTerminated"]
    adapt_curr_algorithms = ["PPOLag"]
    folder_base = "incremental_adaptive_curriculum"
    curr_changes = [10, 20, 40, 100, 300, 700]
    seeds = [5905, 7337, 572, 5689, 3968, 175, 4678, 9733, 3743, 7596]
    betas = [0.5, 1.0, 1.5]
    kappas = [5, 10, 20]

    on_server = torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', dest='exp', type=int, help='Choose experiment', default=0)
    args = parser.parse_args()
    exp = args.exp

    if exp == 3:
        folder_base = "incremental_adaptive_curriculum"
        # Repeat experiments
        wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        seeds = [5905, 7337, 572, 5689, 3968]
        for end_task in range(0, len(curr_changes) + 1):
            with Pool(8) as p:
                args_curr = list(product(curr_algorithms, [end_task], ["curriculum"], seeds, [exp], [1.0], [20]))
                args_adapt_curr = list(product(curr_algorithms, [end_task], ["adaptive_curriculum"], seeds, [exp], [1.0], [20]))
                p.starmap(use_params, args_curr + args_adapt_curr)
    elif exp == 4:
        folder_base = "incremental_adaptive_curriculum"
        # Repeat experiments
        wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        seeds = [175, 4678, 9733, 3743, 7596]
        for end_task in range(0, len(curr_changes) + 1):
            with Pool(8) as p:
                args_curr = list(product(curr_algorithms, [end_task], ["curriculum"], seeds, [exp], [1.0], [20]))
                args_adapt_curr = list(product(curr_algorithms, [end_task], ["adaptive_curriculum"], seeds, [exp], [1.0], [20]))
                p.starmap(use_params, args_curr + args_adapt_curr)


    train_df = pd.read_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    train_df['end_task'] = train_df['end_task'].astype(str)
    train_df['seed'] = train_df['seed'].astype(str)
    train_df = train_df[train_df["end_task"] != "0"]
    eval_df = pd.read_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    eval_df['end_task'] = eval_df['end_task'].astype(str)
    eval_df['seed'] = eval_df['seed'].astype(str)
    eval_df = eval_df[eval_df["end_task"] != "0"]

    # Plot the results
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=train_df)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=eval_df)
    # print_incremental_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)

    # train_df["type"] = train_df["type"].replace("adaptive_curriculum", "adaptive curriculum")
    # eval_df["type"] = eval_df["type"].replace("adaptive_curriculum", "adaptive curriculum")
    # train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    # eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")

    # fig = plt.figure(figsize=(18, 6), dpi=200)
    # additional_folder = ""
    # additional_file_text = ""

    # # Define your dataframes
    # dataframes = [train_df[(train_df["end_task"] == "T") & (train_df["type"] == "adaptive curriculum")], 
    #               eval_df[(eval_df["end_task"] == "T") & (eval_df["type"] == "adaptive curriculum")]]

    # # Create a FacetGrid
    # sns.set_style("whitegrid")
    # g = sns.FacetGrid(dataframes[0], hue='Algorithm', col="Algorithm", col_wrap=4, height=4)  # col_wrap controls how many plots per row

    # # Map the lineplot onto the FacetGrid
    # g.map_dataframe(sns.lineplot, x='step', y="current_task", errorbar="sd" if False else "se", estimator=None, units='seed', alpha=0.4)

    # # Adjust the plot (optional)
    # g.set_axis_labels("x1000 Steps", "Current Task", fontsize=14)
    # g.set_titles("{col_name}", size=14)

    # # Force the x-axis ticks and labels to show on all plots
    # for ax in g.axes.flatten():
    #     ax.tick_params(labelbottom=True)

    # # plt.title(f"Performance of adaptive agents using different betas and kappas")
    # fig.suptitle("Performance of adaptive agents using different betas and kappas")
    # plt.tight_layout(pad=2)
    # if not os.path.isdir(f"figures/{folder_base}/{additional_folder}"):
    #     os.makedirs(f"figures/{folder_base}/{additional_folder}")
    # plt.savefig(f"figures/{folder_base}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}current_task.png")
    # plt.savefig(f"figures/{folder_base}/{additional_folder + '/' if additional_folder != '' else ''}{additional_file_text}current_task.pdf")
    # plt.close()
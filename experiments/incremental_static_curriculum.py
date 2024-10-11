import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *

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

    # # Repeat experiments
    # wandb.login(key="4735a1d1ff8a58959d482ab9dd8f4a3396e2aa0e")
    # for end_task in range(1, len(curr_changes) + 1):
    #     with Pool(8) as p:
    #         args_base = list(product(baseline_algorithms, [end_task], ["baseline"], seeds))
    #         args_curr = list(product(curr_algorithms, [end_task], ["curriculum"], seeds))
    #         args = args_curr + args_base
    #         p.starmap(use_params, args)

    train_df = pd.read_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    train_df['end_task'] = train_df['end_task'].astype(str)
    train_df['seed'] = train_df['seed'].astype(str)
    eval_df = pd.read_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    eval_df['end_task'] = eval_df['end_task'].astype(str)
    eval_df['seed'] = eval_df['seed'].astype(str)

    # print(eval_df[(eval_df["end_task"] == "1") & (eval_df["type"] == "baseline")])

    # Plot the results
    train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=train_df)
    eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=eval_df)
    # print_incremental_eval(folder=folder_base, train_df=train_df, eval_df=eval_df, save_freq=save_freq, cost_limit=cost_limit)

    # train_df.to_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    # eval_df.to_csv(f"./figures/{folder_base}/comparison/eval_df.csv")

    # train_df = pd.read_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    # eval_df = pd.read_csv(f"./figures/{folder_base}/comparison/eval_df.csv")


    folder_base = "incremental_static_curriculum_ablation"

    train_df = pd.read_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    train_df['end_task'] = train_df['end_task'].astype(str)
    train_df['seed'] = train_df['seed'].astype(str)
    eval_df = pd.read_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    eval_df['end_task'] = eval_df['end_task'].astype(str)
    eval_df['seed'] = eval_df['seed'].astype(str)

    # Plot the results
    # train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=train_df)
    # eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=eval_df)


    folder_base = "incremental_static_curriculum_lag_init"

    train_df = pd.read_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    train_df['end_task'] = train_df['end_task'].astype(str)
    train_df['seed'] = train_df['seed'].astype(str)
    eval_df = pd.read_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    eval_df['end_task'] = eval_df['end_task'].astype(str)
    eval_df['seed'] = eval_df['seed'].astype(str)

    # Plot the results
    # train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=train_df)
    # eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=eval_df)
    
    
    folder_base = "incremental_static_curriculum_r"

    train_df = pd.read_csv(f"./figures/{folder_base}/comparison/train_df.csv")
    train_df['end_task'] = train_df['end_task'].astype(str)
    train_df['seed'] = train_df['seed'].astype(str)
    eval_df = pd.read_csv(f"./figures/{folder_base}/comparison/eval_df.csv")
    eval_df['end_task'] = eval_df['end_task'].astype(str)
    eval_df['seed'] = eval_df['seed'].astype(str)

    # Plot the results
    # train_df = plot_incremental_train(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=train_df)
    # eval_df = plot_incremental_eval(folder=folder_base, curr_changes=curr_changes, cost_limit=cost_limit, combined_df=eval_df)
    
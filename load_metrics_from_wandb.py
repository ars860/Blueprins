import matplotlib.pyplot as plt
import wandb


def load_metrics_from_wandb(proj, run_name):
    api = wandb.Api()

    run = api.run(f"ars860/{proj}/{run_name}")
    for i, row in run.history().iterrows():
        print(row["train_loss"], row["test_loss"], row["test_iou"])

    # plt.plot([row["train_loss"] for i, row in run.history().iterrows()], label="train loss")
    # plt.plot([row["test_loss"] for i, row in run.history().iterrows()], label="test loss")
    plt.plot([row["test_iou"] for i, row in run.history().iterrows()], label="test iou")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    load_metrics_from_wandb('diplom_segmentation', "w7274q9w")
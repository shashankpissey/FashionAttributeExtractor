import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def visualise_all_class_pipeline_f1(path):
    df = pd.read_csv(path)
    pipelines =  sorted(df["pipeline"].unique())
    metrics_plot = ["F1_IoU50", "mean_iou", "mean_dice"]
    num = len(metrics_plot)
    num_cols = len(pipelines)

    fig, axes = plt.subplots(num, num_cols, figsize = (20,16), sharex=True)

    for rw, metric in enumerate(metrics_plot):
        for ax, pipeline in enumerate(pipelines):
            ax_in = axes[rw, ax]
            filtered_df = df[df["pipeline"] == pipeline]

            sns.barplot(filtered_df, x="class", y= metric, hue="class", palette="Greens", ax=ax_in)

            for cont in ax_in.containers:
                ax_in.bar_label(cont, fmt='%.2f', padding = 2, fontsize=8)
            ax_in.set_ylim(0,1.2)

            if rw == 0:
                ax_in.set_title(pipeline, fontsize= 12, fontweight="bold", pad=12)
            else:
                ax_in.set_title("")

            if ax == 0:
                ax_in.set_ylabel(metric.upper(), fontsize=12, fontweight="bold", labelpad=10)
            else:
                ax_in.set_ylabel("")
            if rw == num - 1:
                ax_in.set_xlabel("Object Class", fontsize= 10, labelpad=10)
                ax_in.tick_params(axis='x', rotation=45)
            else:
                ax_in.set_xlabel("")
    save_path = os.path.dirname(path)+"/class_level_f1_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved in {save_path}")


def visualise_all_pipeline_f1(path):
    df = pd.read_csv(path)
    metrics_plot = ["F1","F1_IoU50", "mean_iou", "mean_dice"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for i, metric in enumerate(metrics_plot):
        row, col =i //2, i% 2
        ax = axes[row, col]

        sns.barplot(data=df, x="class", y=metric, hue="pipeline", ax=ax, palette="Set1")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric.capitalize()} by Class and Pipeline", fontsize=12)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="center", fontsize=10, fontweight="bold", rotation=90)
        if i == 3:
            ax.legend(title="Pipeline", loc="lower right")
        else:
            ax.legend_.remove()
    plt.tight_layout()
    save_path = os.path.dirname(path)+f"/class_level_pipeline_{metric}_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

def generate_result_table(path):
    metric_map = {"F1": "F1", "Precision@IoU50": "precision_IoU50", "Recall@IoU50": "recall_IoU50", "F1@IoU50": "F1_IoU50", "MeanIoU": "mean_iou", "MeanDice": "mean_dice"}
    df = pd.read_csv(path)
    result_df = df.groupby(["class", "pipeline"]).agg({
        col_name: "mean" for col_name in metric_map.values()
    }).rename(columns={v:k for k, v in metric_map.items()})

    result_df_pipeline = df.groupby(["pipeline"]).agg({
            col_name: "mean" for col_name in metric_map.values()
        }).rename(columns={v:k for k, v in metric_map.items()})

    result_df.to_csv(os.path.dirname(path)+f"/result_table.csv")
    result_df_pipeline.to_csv(os.path.dirname(path)+f"/result_df_pipeline.csv")
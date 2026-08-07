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
    metrics_plot = ["F1_IoU50", "mean_iou", "mean_dice"]
    for metric in metrics_plot:
        pivot_df = df.pivot(index="class", columns="pipeline", values=metric).sort_index()

        width = 0.25
        x = np.arange(len(pivot_df.index))
        colors = {"Pipeline_A": "#B4E3B3", "Pipeline_B": "#319C43", "Pipeline_C": "#1B5E20"}
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, pipeline in enumerate(pivot_df.columns):
            ax.bar(x + (i - 1) * width, pivot_df[pipeline], width, label=pipeline, color=colors.get(pipeline))

        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index, rotation=0, fontsize=10)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric.capitalize()} by Class and Pipeline", fontsize=12, fontweight="bold")
        ax.legend(title="Pipeline", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_path = os.path.dirname(path)+f"/class_level_pipeline_{metric}_plot.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
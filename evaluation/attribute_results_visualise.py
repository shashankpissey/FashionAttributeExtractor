import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from data.attributes_list import ATTRIBUTE_MAPPING


def load_allfiles(paths):
    all_summary_files = []
    all_class_files = []
    all_master_files = []
    for types in paths:
        df = pd.read_csv(types["summary_file_path"])
        df["model"] = types["model"]
        df["pipeline"] = types["Pipeline"]
        all_summary_files.append(df)
        df1 = pd.read_csv(types["per_class_path"])
        df1["model"] = types["model"]
        df1["pipeline"] = types["Pipeline"]
        all_class_files.append(df1)
        df2 = pd.read_csv(types["master_path"])
        df2["model"] = types["model"]
        df2["pipeline"] = types["Pipeline"]
        all_master_files.append(df2)
    summary_df = pd.concat(all_summary_files, ignore_index=True)
    class_df = pd.concat(all_class_files, ignore_index=True)
    master_df = pd.concat(all_master_files, ignore_index=True)
    return summary_df, class_df, master_df

def build_category_level_metrics(class_df):
    """
        This method builds unweighted F1-score acorss attribute tagging
    """
    categories_df = class_df[class_df["category"] != "Overall (weighted) metric"]
    print(class_df.groupby(["pipeline", "model", "category"]).size())

    pivot_categories = pd.pivot_table(
        categories_df,
        values="f1-score",
        index="category",
        columns=[ "pipeline", "model"],
        aggfunc="mean"
    ).round(3)
    support = categories_df.groupby("category")["support"].first().astype(int)
    pivot_categories.insert(0, "Support", support)

    return pd.DataFrame(pivot_categories)

def build_attribute_level_metrics(summary_df, target_combinations):
    """
        Attribute level Weighted F1
    """
    target_category = [] 
    target_attributes = []

    for cat, att in target_combinations.items():
        target_category.append(cat)
        target_attributes.append(att)

    filtered_df = summary_df[(summary_df["category"].isin(target_category)) & (summary_df["attribute"].isin(target_attributes))]

    att_table = pd.pivot_table(
        filtered_df,
        values="macro_f1",
        index=["category", "attribute"],
        columns=["model", "pipeline"],
        aggfunc="mean"
    ).round(3)

    return att_table

def get_weighted_average(g):
    if g["support"].sum() == 0:
        return 0
    weighted_avg = (g["macro_f1"] * g["support"]).sum() / g["support"].sum()
    return weighted_avg


def build_macro_avg_per_category(summary_df):
    """
        Category Level Macro-F1
    """
    grouped_df = summary_df.groupby(["category", "model", "pipeline"])
    weighted_df = grouped_df.apply(get_weighted_average).reset_index(name="weighted_f1")
    category_weighted = pd.pivot_table(
        weighted_df,
        values="weighted_f1",
        index="category",
        columns=["model", "pipeline"]
    ).round(3)
    single_run = summary_df[(summary_df["model"] == "CLIP") & (summary_df["pipeline"] == "Pipeline B")]
    total_support_series = single_run.groupby("category")["support"].first().astype(int)
    category_weighted.insert(0, "Total_Support", total_support_series)

    return category_weighted

def build_weighted_avg_per_attribute(summary_df):
    """
        This gives weighted F1 per attribute type.
    """
    grouped_df = summary_df.groupby(["attribute", "model", "pipeline"])
    print("Grouped")
    print(grouped_df.head())
    print("Grouped")
    weighted_df = grouped_df.apply(get_weighted_average).reset_index(name="weighted_f1")
    attribute_weighted = pd.pivot_table(
        weighted_df,
        values="weighted_f1",
        index="attribute",
        columns=["model", "pipeline"]
    ).round(3)
    single_run = summary_df[(summary_df["model"] == "CLIP") & (summary_df["pipeline"] == "Pipeline A")]
    total_support_series = single_run.groupby("attribute")["support"].sum().astype(int)
    attribute_weighted.insert(0, "Total_Support", total_support_series)

    return attribute_weighted

def build_pipeline_overall_summary(summary_df):
    cat_level_aggregate = (summary_df.groupby(["model", "pipeline", "category"]).agg(mean_macro_f1=("macro_f1", "mean"), support=("support", "first")).reset_index())

    final_overall = []
    for (model, pipeline), g in cat_level_aggregate.groupby(["model", "pipeline"]):
        final_overall.append({
            "model": model,
            "pipeline": pipeline,
            "unweighted_macro_f1": round(g["mean_macro_f1"].mean(),3),
            "weighted_macro_f1": round(np.average(g["mean_macro_f1"], weights=g["support"]),3)
        })
    final_df = pd.DataFrame(final_overall)
    return final_df

def is_exact_match(row, col_list):
    for pred_col, true_col in col_list:
        pred_val = str(row[pred_col]).lower().strip().replace("-", "_").replace(" ", "_")
        true_val = str(row[true_col]).lower().strip().replace("-", "_").replace(" ", "_")
        if pred_val != true_val:
            return False
    return True


def build_master_accuracy(group_df, category):
    category_df = group_df[group_df["true_category"].str.lower() == category.lower()].copy()
    if len(category_df) == 0:
        return None

    cols_to_compare = []
    for pred_col, true_col in ATTRIBUTE_MAPPING.items():
        if pred_col == "predicted_base_style":
            continue
        if pred_col not in category_df.columns or true_col not in category_df.columns:
            continue
        if category in ["top", "outer_top"]:
            if pred_col in ["predicted_waist_definition", "predicted_rise", "predicted_leg_shape"]:
                continue
        if category in ["jeans", "shorts", "skirt", "trousers"]:
            if pred_col in ["predicted_sleeve_length", "predicted_sleeve_volume", "predicted_neckline"]:
                continue
        if category in ["bag_wallet", "shoes"]:
            if pred_col not in ["predicted_opacity_level", "predicted_closure_mode", "predicted_visual_weight", "predicted_item_type", "predicted_material", "predicted_colour"]:
                continue
        cols_to_compare.append((pred_col, true_col))

    category_df["exact_row_match"] = category_df.apply(lambda row: is_exact_match(row, cols_to_compare), axis=1)
    tp = category_df["exact_row_match"].sum()
    total = len(category_df)
    final_df = {
        "category": category,
        "exact_match_accuracy": round(tp/total, 3) if total > 0 else 0,
        "TP": tp,
        "FP": (total - tp),
        "total_instance": total
    }
    return final_df

def get_master_accuracy(master_df):
    results = []
    unique_categories = master_df["true_category"].dropna().unique()
    grouped_master = master_df.groupby(["model", "pipeline"])

    for (model, pipeline), group_df in grouped_master:
        for category in unique_categories:
            metrics = build_master_accuracy(group_df, category)
            if metrics:
                results.append({
                    "category": category,
                    "model": model,
                    "pipeline": pipeline,
                    "exact_match_accuracy": metrics["exact_match_accuracy"],
                    "total_instance": metrics["total_instance"]
                })
    if not results:
        return pd.DataFrame(results)

    results_df = pd.DataFrame(results)

    pivot_details = pd.pivot_table(
        results_df,
        values="exact_match_accuracy",
        index="category",
        columns=["model", "pipeline"]
    ).round(3)

    support = results_df.groupby("category")["total_instance"].first()
    pivot_details.insert(0, "Support", support)
    return pivot_details


def visualise_attribute_type(data_path, base_path):
    data = pd.read_csv(data_path, skiprows=3, header=None)
    data.columns = [
    "attribute", "Total Support",
    "CLIP_A", "CLIP_B", "CLIP_C", 
    "SigLip_A", "SigLip_B", "SigLip_C"
    ]
    print(data)
    data["attribute"] = data["attribute"].str.replace("predicted_","",)
    score_cols = ["CLIP_A", "CLIP_B", "CLIP_C", "SigLip_A", "SigLip_B", "SigLip_C"]
    for col in score_cols:
        data[col] = pd.to_numeric(data[col])
    cmap = plt.colormaps["Set1"]
    colors = [cmap(0), cmap(1), cmap(2)]
    attributes = data["attribute"].tolist()
    print(attributes)
    x = np.arange(len(attributes))
    width = 0.25
    plt.figure(figsize=(12,8))
    plt.bar(x-width, data["CLIP_A"], width, label="Pipeline A", color=colors[0])
    plt.bar(x, data["CLIP_B"], width, label="Pipeline B", color=colors[1])
    plt.bar(x + width, data["CLIP_C"], width, label="Pipeline C", color=colors[2])
    plt.title("CLIP Model", fontsize=18)
    plt.xticks(x, attributes, rotation=45, ha="right", fontsize=10)
    plt.ylim(0,1.0)
    plt.grid(axis="y", linestyle="-", alpha=0.5)
    plt.tick_params(axis="y", length=0)
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80))
    plt.tight_layout()
    plt.savefig(f"{base_path}/attributes_graph_CLIP.png", dpi=300)

    plt.figure(figsize=(12,8))
    plt.bar(x-width, data["SigLip_A"], width, label="Pipeline A", color=colors[0])
    plt.bar(x, data["SigLip_B"], width, label="Pipeline B", color=colors[1])
    plt.bar(x + width, data["SigLip_C"], width, label="Pipeline C", color=colors[2])
    plt.title("SigLip Model", fontsize=18)
    plt.xticks(x, attributes, rotation=45, ha="right", fontsize=10)
    plt.ylim(0,1.0)
    plt.grid(axis="y", linestyle="-", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80))
    plt.tight_layout()
    plt.savefig(f"{base_path}/attributes_graph_SigLip.png", dpi=300)

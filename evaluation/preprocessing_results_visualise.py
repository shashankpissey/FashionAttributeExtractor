import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from data.attributes_list import ATTRIBUTE_MAPPING


def load_allfiles(paths):
    all_summary_files = []
    all_class_files = []
    all_master_files = []
    for types in paths:
        df = pd.read_csv(types["summary_file_path"])
        df["Background Context"] = types["Background Context"]
        df["Prompt Method"] = types["Prompt Method"]
        all_summary_files.append(df)
        df1 = pd.read_csv(types["per_class_path"])
        df1["Background Context"] = types["Background Context"]
        df1["Prompt Method"] = types["Prompt Method"]
        all_class_files.append(df1)
        df2 = pd.read_csv(types["master_path"])
        df2["Background Context"] = types["Background Context"]
        df2["Prompt Method"] = types["Prompt Method"]
        all_master_files.append(df2)
    summary_df = pd.concat(all_summary_files, ignore_index=True)
    class_df = pd.concat(all_class_files, ignore_index=True)
    master_df = pd.concat(all_master_files, ignore_index=True)
    return summary_df, class_df, master_df

def build_category_level_metrics(class_df):
    categories_df = class_df[class_df["category"] != "Overall (weighted) metric"]
    # print(class_df.groupby(["pipeline", "model", "category"]).size())

    pivot_categories = pd.pivot_table(
        categories_df,
        values="f1-score",
        index="category",
        columns=[ "Background Context", "Prompt Method"],
        aggfunc="mean"
    ).round(3)
    support = categories_df.groupby("category")["support"].first().astype(int)
    pivot_categories.insert(0, "Support", support)

    return pd.DataFrame(pivot_categories)

def build_attribute_level_metrics(summary_df, target_combinations):
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
        columns=["Background Context", "Prompt Method"],
        aggfunc="mean"
    ).round(3)

    return att_table

def get_weighted_average(g):
    if g["support"].sum() == 0:
        return 0
    weighted_avg = (g["macro_f1"] * g["support"]).sum() / g["support"].sum()
    return weighted_avg

def build_weighted_avg_per_attribute(summary_df):
    grouped_df = summary_df.groupby(["attribute",  "Background Context", "Prompt Method"])
    # print("Grouped")
    # print(grouped_df.head())
    # print("Grouped")
    weighted_df = grouped_df.apply(get_weighted_average).reset_index(name="weighted_f1")
    attribute_weighted = pd.pivot_table(
        weighted_df,
        values="weighted_f1",
        index="attribute",
        columns=["Background Context", "Prompt Method"]
    ).round(3)
    single_run = summary_df[(summary_df["Background Context"] == "segmented_b") & (summary_df["Prompt Method"] == "Direct")]
    support = single_run.groupby("attribute")["support"].sum().astype(int)
    attribute_weighted.insert(0, "Total_Support", support)

    return attribute_weighted

def build_pipeline_overall_summary(summary_df):
    cat_level_aggregate = (summary_df.groupby(["Background Context", "Prompt Method", "category"]).agg(mean_macro_f1=("macro_f1", "mean"), support=("support", "first")).reset_index())

    final_overall = []
    for (preprocessing, method), g in cat_level_aggregate.groupby(["Background Context", "Prompt Method"]):
        final_overall.append({
            "Background Context": preprocessing,
            "pipeline": method,
            "unweighted_macro_f1": round(g["mean_macro_f1"].mean(),3),
            "weighted_macro_f1": round(np.average(g["mean_macro_f1"], weights=g["support"]),3),
            "Support": int(g["support"].sum())
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
    grouped_master = master_df.groupby(["Background Context", "Prompt Method"])

    for (preprocessing, method), group_df in grouped_master:
        for category in unique_categories:
            metrics = build_master_accuracy(group_df, category)
            if metrics:
                results.append({
                    "category": category,
                    "Background Context": preprocessing,
                    "Prompt Method": method,
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
        columns=["Background Context", "Prompt Method"]
    ).round(3)

    support = results_df.groupby("category")["total_instance"].first()
    pivot_details.insert(0, "Support", support)
    return pivot_details


def visualise_preprocessing_asheatmap(basepath, datapath):
    data = pd.read_csv(basepath+"/"+datapath, skiprows=3, header=None)
    data.columns = [
        "attribute", "Total Support", "Full Dimmed\n(Descriptive)", "Full Dimmed\n(Direct)",
        "Contextual Dimmed\n(Descriptive)", "Contextual Dimmed\n(Direct)",
        "Tight Crop\n(Descriptive)", "Tight Crop\n(Direct)"
        ]
    data["attribute"] = data["attribute"].str.replace("predicted_","",)
    data.set_index("attribute", inplace=True)
    data = data.drop(columns=["Total Support"])
    order = ["Full Dimmed\n(Direct)", "Contextual Dimmed\n(Direct)", "Tight Crop\n(Direct)", "Full Dimmed\n(Descriptive)", "Contextual Dimmed\n(Descriptive)", "Tight Crop\n(Descriptive)"]
    data = data[order]
    descriptive_attributes = ["item_type", "silhouette", "print_type", "formality_band", "seasonality", "closure_mode", "drape", "waist_definition", "texture", "leg_shape", "length"]
    data.index = [f"{attr}*" if attr in descriptive_attributes else attr for attr in data.index]

    data_diff_df = pd.DataFrame({
        "Full Dimmed Mean Average Difference": np.abs(data["Full Dimmed\n(Descriptive)"] - data["Full Dimmed\n(Direct)"]),
        "Contextual Dimmed Mean Average Difference": np.abs(data["Contextual Dimmed\n(Descriptive)"] - data["Contextual Dimmed\n(Direct)"]),
        "Tight Crop Mean Average Difference": np.abs(data["Tight Crop\n(Descriptive)"] - data["Tight Crop\n(Direct)"])
    })
    data_diff_df = data_diff_df.reset_index()

    plt.figure(figsize=(12,8))
    ax = sns.heatmap(
        data,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        cbar_kws={"label": "Metric Score"}
    )

    plt.axvline(x=3, color="white", linewidth=5)
    plt.title("Attribute Extraction Weighted-F1 across all categories\nfor 2 prompting methods and 3 Background Context\n", fontsize=11)
    plt.ylabel("Fashion Attributes", fontsize=12)
    plt.xlabel("Background Context", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{basepath}/attribute_visualise_preprocessing.png", dpi=300)
    plt.close()

    print(data_diff_df)
    data_diff_df["grouping_desc"] = ["Descriptive (*)" if "*" in attr else "Direct" for attr in data_diff_df["index"]]
    mean_values = data_diff_df.groupby("grouping_desc").mean(numeric_only=True).round(3)
    mean_df = pd.DataFrame(mean_values)
    mean_df.to_csv(f"{basepath}/MAD.csv")

    df_visuliase = data_diff_df.reset_index().melt(
        id_vars=["index", "grouping_desc"],
        value_vars=[
            "Full Dimmed Mean Average Difference",
            "Contextual Dimmed Mean Average Difference",
            "Tight Crop Mean Average Difference"
        ],
        var_name="Context",
        value_name="Mean Average Difference"
    )
    df_visuliase["Context"] = df_visuliase["Context"].str.replace(" Mean Average Difference", "")
    df_visuliase = df_visuliase.sort_values(by="grouping_desc")
    plt.figure(figsize=(12,8))
    sns.barplot(data=df_visuliase, x="index", y="Mean Average Difference", hue="Context", palette="Set1")
    plt.title("Mean Average Difference (Direct vs Descriptive) per attribute per context")
    plt.ylabel("Mean Average Difference (Weighted-F1 Score)")
    plt.xlabel("Attributes")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Background Context")
    plt.tight_layout()
    plt.savefig(f"{basepath}/mean_average_difference.png", dpi=300)

def routingaccuracy(merged_df, routing_column="predicted_base_style"):
    filtered = ["upper_body", "dress", "pants"]
    df = merged_df[merged_df["main_category"].isin(filtered)]
    contexts = df["Background Context"].unique()

    for context in contexts:
        df_context = df[df["Background Context"] == context]
        df_direct = df_context[df_context["Prompt Method"] == "Direct"]
        df_descriptive = df_context[df_context["Prompt Method"] == "Descriptive"]

        df_merged = pd.merge(
            df_direct[['filename', routing_column]], 
            df_descriptive[['filename', routing_column]], 
            on='filename', 
            suffixes=('_dir', '_desc')
        )

        col_dir = f"{routing_column}_dir"
        col_desc = f"{routing_column}_desc"

        mismatches = df_merged[col_dir] != df_merged[col_desc]
        total_images = len(df_merged)
        rerouted_images = mismatches.sum()
        if total_images > 0:
            percentage = (rerouted_images / total_images) * 100
            print(f"\nBackground Context: {context}")
            print(f"Total images evaluated: {total_images}")
            print(f"Images re-routed: {rerouted_images}")
            print(f"Percentage re-routed: {percentage:.2f}%")



def generate_confusion_matrix(basepath, gt, pred, master_df):
    master_df = master_df.replace('anbove_knee', 'above_knee')
    tight_crop_df = master_df[master_df["Background Context"] == "segmented_b"]

    direct_df = tight_crop_df[tight_crop_df["Prompt Method"] == "Direct"]
    descriptive_df = tight_crop_df[tight_crop_df["Prompt Method"] == "Descriptive"]

    direct_df = direct_df.dropna(subset=[gt, pred])
    descriptive_df = descriptive_df.dropna(subset=[gt, pred])
    labels = sorted(list(set(direct_df[gt]) | set(direct_df[pred]) | set(descriptive_df[gt]) | set(descriptive_df[pred])))

    print(direct_df[gt])
    print(descriptive_df[pred])

    cm_direct = confusion_matrix(direct_df[gt], direct_df[pred], labels = labels)
    cm_desc = confusion_matrix(descriptive_df[gt], descriptive_df[pred], labels = labels)

    fig, axes = plt.subplots(1,2, figsize=(12,8))
    sns.heatmap(cm_direct, annot=True, fmt="g", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title("Direct Prompt (Tight Crop)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    sns.heatmap(cm_desc, annot=True, fmt="g", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title("Descriptive Prompt (Tight Crop)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{basepath}/confusion_matrix_length.png", dpi=300)
    # plt.show()
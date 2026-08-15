import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)

from data.attributes_list import ATTRIBUTE_MAPPING



class Evaluate_attribute_metrics:

    def __init__(self, gt_file_path, pred_file_path):
        """
        Define all the mapping and files for evaluation
        """
        self.gt_df = pd.read_csv(gt_file_path)
        self.pred_df = pd.read_csv(pred_file_path)
        self.pred_mapping = {
            "all-in-one": "all_in_one"
            }
        self.gt_mapping_clean = {
            "bag / small leather goods": "bag_wallet", 
            "outerwear": "outer_top"
            }
        self.map_gt_to_segment = {
            "skirt": "skirt",
            "trousers": "pants",
            "jeans": "pants",
            "shorts": "pants",
            "top": "top",
            "outer_top": "outer_top",
            "dress": "dress",
            "all_in_one": "dress",
            "bag_wallet": "bag_wallet",
            "shoes": "shoes"
            }

    def extract_base_id_pred(self, filename):
        # This cleans the file name to extract the base_id. It is the actual name of file from source
        if pd.isna(filename):
            return None
        return filename.split("_p0_")[0] if "_p0_" in filename else os.path.splitext(filename)[0]
    
    def extract_segment_type(self, filename):
        # This is used to obtain the segment type from upstream. For GT it uses category and for the prediction it takes from filename that the segmentation has tagged.
        if pd.isna(filename):
            return None
        return os.path.splitext(filename.split("_p0_")[1])[0]
    
    def extract_base_id_gt(self, filename):
        # Used to get the basename of the file for source
        if pd.isna(filename):
            return None
        return os.path.splitext(str(filename))[0]    

    def force_clean_string(self,val):
        # Cleaning the characters in category to bring in standard names so that it can be used for comparison
        if pd.isna(val): 
            return "missing"
        return str(val).lower().strip().replace("-", "_").replace(" ", "_")

    
    def extract_base_id_from_filename(self):
        """
        This method is used to call the methods to prepare the comparison dataframe to obtain a main master dataframe for comparison
        """

        self.pred_df["base_id"] = self.pred_df["filename"].apply(self.extract_base_id_pred)
        self.pred_df["segment_type"] = self.pred_df["filename"].apply(self.extract_segment_type)
        self.pred_df["pred_category"] = self.pred_df["main_category"].apply(self.force_clean_string)
        self.gt_df["base_id"] = self.gt_df["source_url"].apply(self.extract_base_id_gt)
        self.gt_df["true_category"] = self.gt_df["category_x"].str.lower().str.strip()


    def clean_and_map_cols(self):
        # Clean the values so that the prediction and the source matches
        out_of_scope_segments = ["headband_head_covering_hair_accessory", "watch", "sunglasses", "hat"]
        self.pred_df = self.pred_df[~self.pred_df["segment_type"].isin(out_of_scope_segments)].copy()

        
        self.pred_df["pred_category"] = self.pred_df["pred_category"].replace(self.pred_mapping)
        self.pred_df["predicted_base_style"] = self.pred_df["predicted_base_style"].replace(self.pred_mapping)

        self.gt_df["true_category"] = self.gt_df["true_category"].replace(self.gt_mapping_clean)

        self.gt_df["segment_type"] = self.gt_df["true_category"].map(self.map_gt_to_segment)

    def get_y_true(self, row):
        # Obtain the rows that are only present in the source and missing from ground truth
        if row["_merge"] == "left_only":
            return "no_gt"
        return row["true_category"]
    
    def get_y_pred(self, row):
        # Obtain the rows hat are only in ground truth and not in prediction
        if row["_merge"] == "right_only":
            return "no_pred"
        val = row.get("predicted_base_style",None)
        if pd.isna(val):
            return row["segment_type"]
        return str(val).lower().replace("-"," ").strip().split()[-1]
    

    def merge_pred_gt_rows(self):
        """
        This methos is used to prepare a master file that can be used later for calculating the metrics
        """

        self.extract_base_id_from_filename()
        self.clean_and_map_cols()


        gt_filtered = self.gt_df.dropna(subset=["segment_type"]).copy()

        self.pred_df["instance_id"] = self.pred_df.groupby(["base_id", "segment_type"]).cumcount()
        self.gt_df["instance_id"] = self.gt_df.groupby(["base_id", "segment_type"]).cumcount()
        gt_filtered["instance_id"] = gt_filtered.groupby(["base_id", "segment_type"]).cumcount()

        master_df = pd.merge(
            self.pred_df, 
            gt_filtered, 
            on=["base_id", "segment_type", "instance_id"], 
            how="outer",
            indicator=True
        )

        has_pred = master_df["_merge"].isin(["left_only", "both"])
        #As skirt, bag_wallet and shoes has no hierarchy mapping base_style will be empty and hence backfill.
        # This makes the basestyle for confirmed categories as 1 
        master_df.loc[has_pred & (master_df["segment_type"] == "skirt"), "predicted_base_style"] = "skirt"
        master_df.loc[has_pred & (master_df["segment_type"] == "bag_wallet"), "predicted_base_style"] = "bag_wallet"
        master_df.loc[has_pred & (master_df["segment_type"] == "shoes"), "predicted_base_style"] = "shoes"

        master_df["y_true"] = master_df.apply(self.get_y_true, axis = 1)
        master_df["y_pred"] = master_df.apply(self.get_y_pred, axis = 1)

        return master_df

    def evaluate(self, save_path, segment_type):
        """
        This is a main method that runs different functions in order to obtain the final metrics.
        """

        master_df = self.merge_pred_gt_rows()
        master_df.to_csv(f"{save_path}{segment_type}_master.csv")
        
    
        y_true = master_df["y_true"]
        y_pred = master_df["y_pred"]

        not_present_tokens = {"no_gt", "no_pred"}

        category = sorted(set(y_true) | set(y_pred) - not_present_tokens)
        labels = category + sorted(not_present_tokens & (set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(15, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, colorbar=True)
        ax.set_title(f"Category Confusion Matrix  (n={len(master_df)})", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(f"{save_path}{segment_type}_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

        score_labels = [l for l in labels if l not in not_present_tokens]

        overall_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        macro_precision = precision_score(y_true=y_true, y_pred=y_pred, labels=score_labels, average="macro", zero_division=0)
        macro_recall = recall_score(y_true=y_true, y_pred=y_pred, labels=score_labels, average="macro", zero_division=0)
        weighted_precision = precision_score(y_true=y_true, y_pred=y_pred, labels=score_labels, average="weighted", zero_division=0)
        weighted_recall = recall_score(y_true=y_true, y_pred=y_pred, labels=score_labels, average="weighted", zero_division=0)

        class_report = classification_report(y_true=y_true, y_pred=y_pred, labels=score_labels, zero_division=0, output_dict=True)

        per_class_report = pd.DataFrame(class_report).T.loc[score_labels, ["precision", "recall", "f1-score", "support"]]
        per_class_report.index.name = "category"
        per_class_report = per_class_report.round(4)

        full_result = pd.DataFrame([{
            "category": "Overall (weighted) metric",
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": class_report["weighted avg"]["f1-score"],
            "support": per_class_report["support"].sum()
        }]).set_index("category")

        metrics_dataframe = pd.concat([per_class_report, full_result])
        metrics_dataframe.to_csv(f"{save_path}{segment_type}_metrics_per_class.csv")

        attribute_results = []
        summary_results = []

        merged_df = master_df[master_df["_merge"] == "both"].copy()

        target_categories = ["jeans", "shoes","all_in_one", "bag_wallet", "skirt", "trousers", "top", "dress", "outer_top", "shorts"]

        for category in target_categories:

            category_dataframe = merged_df[merged_df["true_category"] == category].copy()

            if len(category_dataframe) == 0:
                continue

            for pred_col, true_col in ATTRIBUTE_MAPPING.items():
                if pred_col not in category_dataframe.columns or true_col not in category_dataframe.columns:
                    continue 

                if pred_col == "predicted_base_style":
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


                y_pred_attr = category_dataframe[pred_col].apply(self.force_clean_string)
                y_true_attr = category_dataframe[true_col].apply(self.force_clean_string)
                print(y_true_attr.unique())
                print(y_pred_attr.unique())
                # print(type(y_true))
				    
                if len(y_true_attr[y_true_attr != "missing"]) == 0:
                    continue

                if len(y_true_attr) > 0:
                    report = classification_report(
                        y_true=y_true_attr,
                        y_pred=y_pred_attr,
                        zero_division=0,
                        output_dict=True
                    )

                    summary_results.append({
                        "category": category,           
                        "attribute": pred_col,
                        "accuracy": report["accuracy"],
                        "macro_precision": report["macro avg"]["precision"],
                        "macro_recall": report["macro avg"]["recall"],
                        "macro_f1": report["macro avg"]["f1-score"],
                        "weighted_precision": report["weighted avg"]["precision"],
                        "weighted_recall": report["weighted avg"]["recall"],
                        "weighted_f1": report["weighted avg"]["f1-score"],
                        "support": report["weighted avg"]["support"]
                    })

                    attr_df = pd.DataFrame(report).T
                    attr_df["category"] = category      
                    attr_df["attribute"] = pred_col 
                    attr_df.index.name = "class_label" 
                    # print(attr_df)
                    
                    classes_only = attr_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
                    attribute_results.append(classes_only)

        if attribute_results:
            final_attr_metrics = pd.concat(attribute_results)
            cols = ["category", "attribute", "precision", "recall", "f1-score","support"]
            final_attr_metrics = final_attr_metrics[cols].round(4)
            final_attr_metrics.to_csv(f"{save_path}{segment_type}_attribute_list.csv")

        
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            cols = ["category", "attribute", "accuracy", "macro_precision", "macro_recall", "macro_f1","weighted_precision","weighted_recall","weighted_f1","support"]
            summary_df = summary_df[cols].round(4)
            summary_df.to_csv(f"{save_path}{segment_type}_summary_metrics.csv", index=False)
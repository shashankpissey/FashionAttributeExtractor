import pandas as pd
import os
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import json

from config import COLOUR_MAPPING

class EvaluateColour:

    def __init__(self, gt_path, pred_path):
        """
        This is constructor method that reads the colour everything that colour mapping is required. Maps the segment 
        """
        with open(COLOUR_MAPPING, "r") as colour_map:
            self.colour_mapping = json.load(colour_map)
            self.colour_mapping = self.colour_mapping["PARENT_COLOUR_MAP"]
        self.pred_csv = pd.read_csv(pred_path)
        self.gt_csv = pd.read_csv(gt_path)
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
        
    def extract_segment_type(self,filename):
        """
        This method maps the segment name like top, outer_top based on the file name from segmentation
        """
        if pd.isna(filename):
            return None
        # print(filename)
        return os.path.splitext(filename.split("_p0_")[1])[0] if "_p0_" in filename else os.path.splitext(filename)[0]
    
    def extract_base_id_pred(self,f):
        """
        Gives the base_id of image which is without the extension and any additional like p0 or category
        """
        if pd.isna(f): return None
        return f.split("_p0_")[0] if "_p0_" in f else os.path.splitext(f)[0]

    def extract_base_id_gt(self, f):
        """
        This gives base_id from ground truth by cleaning the extension
        """
        if pd.isna(f): return None
        return os.path.splitext(str(f))[0]
    
    def evaluate_primary_colour(self, pred_col, gt_col, output_name, save_path):
        """
        The main method that is used for colour evaluation. It reads the prediction and ground truth. Merges based on base_id from file name and segment type based on category and filename. Checks the colorur mapping accuracy and creates a confusion matrix
        """
        self.pred_csv["base_id"] = self.pred_csv["filename"].apply(self.extract_base_id_pred)
        self.pred_csv["segment_type"] = self.pred_csv["filename"].apply(self.extract_segment_type)

        self.gt_csv["base_id"] = self.gt_csv["source_url"].apply(self.extract_base_id_gt)
        self.gt_csv["true_category"] = self.gt_csv["category_x"].str.lower().str.strip()
        self.gt_csv["segment_type"] = self.gt_csv["true_category"].map(self.map_gt_to_segment)

        self.pred_csv["instance_id"] = self.pred_csv.groupby(["base_id", "segment_type"]).cumcount()

        gt_filtered= self.gt_csv.dropna(subset=["segment_type"]).copy()
        gt_filtered["instance_id"] = gt_filtered.groupby(["base_id", "segment_type"]).cumcount()

        merged_df = pd.merge(self.pred_csv,gt_filtered,on=["base_id", "segment_type", "instance_id"], how="inner")

        merged_df = merged_df[merged_df["role_x"].isin(["bag", "outerwear", "bottom", "footwear","top","one_piece"])]

        valid = merged_df[gt_col].notna() & merged_df[pred_col].notna()
        merged_df = merged_df[valid]

        y_true = merged_df[gt_col]
        y_pred = merged_df[pred_col]
        # y_true=y_true.map(self.colour_mapping).fillna(y_true)
        # y_pred=y_pred.map(self.colour_mapping).fillna(y_pred)
        # print(y_true)

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        print(f"\n Top 1 Primary Color Evaluation")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        df = pd.DataFrame(report).transpose()
        df.to_csv(f"report{output_name}.csv")

        labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(15, 12))
        disp.plot(ax=ax, xticks_rotation=90, colorbar=False, cmap=plt.cm.Blues)
        ax.set_title(f"Primary Color — Confusion Matrix (n={len(merged_df)})")
        plt.tight_layout()
        plt.savefig(f"{output_name}_confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Saved: {output_name}_confusion_matrix.png")

        return {
            "accuracy": round(acc, 4),
            "report": report
        }





from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO

from utils.evaluation_config import EvaluatePipeline
from utils.log_config import get_logger

logger = get_logger(__name__)

class Evaluate_IoU:

    def __init__(self):
        self.eval_config = EvaluatePipeline()

    def evaluate_pipeline_iou(self, coco, category, mask_dir, pipeline_name, target_filenames):
        """
        This method is used to evaluate the IoU for segmentation
        It reads the COCO file merge the masks using logical or
        It takes the prediction mask from the saved location and then finds the intersection, union to find the IoU.
        It also maps the TP, FP, FN (presence) This only gives if there are masks in prediction and if there are masks in the ground truth dataset and counts as TP. It gives just presence metrics
        """
        results = []

        file_to_take = set(target_filenames)
        img_ids = []
        for img_id in coco.getImgIds():
            filename = coco.loadImgs(img_id)[0]["file_name"]
            if not filename.endswith("_p0.png"):
                continue
            if not filename in file_to_take:
                continue
            img_ids.append(img_id)
        # img_ids = [img_id for img_id in coco.getImgIds() if coco.loadImgs(img_id)[0]["file_name"].endswith("_p0.png")]

        for image_id in img_ids:
            image_file = coco.loadImgs(image_id)[0]
            filename = image_file["file_name"]
            filename_stem = Path(filename).stem
            h, w = image_file["height"], image_file["width"]

            for cls in self.eval_config.TARGET_CLASS:
                mapped_true_names = self.eval_config.GT_MAPPING[cls]
                category_ids = [category[name] for name in mapped_true_names if name in category]

                annotation_ids = coco.getAnnIds(imgIds=[image_id], catIds=category_ids)
                gt_present = len(annotation_ids) > 0

                file_suffix = self.eval_config.MASK_FILE_NAMES[cls]
                mask_path = mask_dir + f"{filename_stem}_{file_suffix}_mask.png"
                mask_path = Path(mask_path)
                prediction_present = mask_path.exists()

                # print(mask_path)
                # mask_path = f"{mask_dir}+14603998101890_never-fully-dressed-cream-mirror-skirt6_p0_bag_mask.png"
                
                if not gt_present and not prediction_present:
                    continue

                iou = np.nan
                dice = np.nan
                detection_type = "TN"

                if gt_present and not prediction_present:
                    detection_type = "FN"
                elif not gt_present and prediction_present:
                    detection_type = "FP"
                elif gt_present and prediction_present:
                    detection_type = "TP"

                    annotations = coco.loadAnns(annotation_ids)
                    true_mask = np.zeros((h, w), dtype=np.uint8)
                    for annotation in annotations:
                        true_mask = np.logical_or(true_mask, coco.annToMask(annotation))

                    prediction_image = Image.open(mask_path).convert("L")
                    prediction_mask = (np.array(prediction_image) > 0).astype(np.uint8)

                    intersection = np.logical_and(true_mask, prediction_mask).sum()
                    union = np.logical_or(true_mask, prediction_mask).sum()
                    iou = float(intersection / union) if union > 0 else 1.0

                    tp = np.logical_and(true_mask == 1, prediction_mask == 1).sum()
                    fp = np.logical_and(true_mask == 0, prediction_mask == 1).sum()
                    fn = np.logical_and(true_mask == 1, prediction_mask == 0).sum()

                    denominatot = (2 * tp + fp + fn)
                    dice = float(2 * tp) / denominatot if denominatot > 0.0 else 1.0

                results.append({
                    "pipeline": pipeline_name,
                    "image": filename,
                    "class": cls,
                    "iou": iou,
                    "iou_50": iou >= 0.5 if not np.isnan(iou) else False,
                    "has_gt": gt_present,
                    "has_pred": prediction_present,
                    "detection_type": detection_type,
                    "dice": dice
                })

        return pd.DataFrame(results)

    def calculate_presence_f1(self, df):
        """
        This method uses the TP, FP, FN from the previous step and calculates the Precision, Recall and F1-Score for the segmentation
        """
        results = []
        for cls in self.eval_config.TARGET_CLASS:
            type_df = df[df["class"] == cls]
            tp = int((type_df["detection_type"] == "TP").sum())
            fp = int((type_df["detection_type"] == "FP").sum())
            fn = int((type_df["detection_type"] == "FN").sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            results.append({
                "class": cls, "TP": tp, "FP": fp, "FN": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "F1": round(f1, 4)
            })

        seg_f1_df = pd.DataFrame(results)
        macro_f1 = round(seg_f1_df["F1"].mean(), 4)
        return seg_f1_df, macro_f1

    def calculate_iou50_f1(self, df, iou_threshold=0.5):
        """
        This methof calculates the precision, recall and f1-score baed on the standard threshold of 0.5. If the IoU is < 0.50 it counts the prediction as FP and FN.
        """
        results = []
        for cls in self.eval_config.TARGET_CLASS:
            type_df = df[df["class"] == cls]
            tp = 0
            fp = 0
            fn = 0

            for _, row in type_df.iterrows():
                if row["has_gt"] and row["has_pred"]:
                    if row["iou"] >= iou_threshold:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif row["has_pred"] and not row["has_gt"]:
                    fp += 1
                elif not row["has_pred"] and row["has_gt"]:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            results.append({
                "class": cls, "TP_IoU50": tp, "FP_IoU50": fp, "FN_IoU50": fn,
                "precision_IoU50": round(precision, 4),
                "recall_IoU50": round(recall, 4),
                "F1_IoU50": round(f1, 4)
            })

        seg_f1_df = pd.DataFrame(results)
        macro_f1 = round(seg_f1_df["F1_IoU50"].mean(), 4)
        return seg_f1_df, macro_f1

    def evaluate_pipeline(self, file_names):
        """
        This is used as a main method that runs through the different functions and finally creates separaate files along with a consolidated file
        """
        coco = COCO(self.eval_config.COCO_JSON_PATH)
        category_to_id = {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}

        all_iou = []
        all_detection_f1 = []
        all_iou50_f1 = []
        macro_f1_pipeline = []
        macro_segmentation = []
        all_iou50 = []

        for pipeline in self.eval_config.PIPELINES:
            pipeline_name = pipeline.split()[0]
            mask_path = self.eval_config.PRED_MASK_PATH + "/" + pipeline_name + "/eval_images/"

            iou_df = self.evaluate_pipeline_iou(coco, category_to_id, mask_path, pipeline_name, file_names)
            iou_df.to_csv(f"{self.eval_config.OUTPUT_DIR}per_image_iou_{pipeline_name}.csv", index=False)

            # Obtain the mean, median IoU and save
            iou_summary = (
                iou_df[iou_df["iou"].notna()]
                .groupby("class")
                .agg(
                    count=("iou", "count"),
                    mean_iou=("iou", "mean"),
                    mean_dice=("dice", "mean"),
                    median_iou=("iou", "median"),
                    median_dice=("dice", "median"),
                    min_iou=("iou", "min"),
                    max_iou=("iou", "max"),
                )
                .sort_values("mean_iou", ascending=False)
            )
            iou_summary.to_csv(f"{self.eval_config.OUTPUT_DIR}class_segmentation_summary_{pipeline_name}.csv")
            iou_summary_reset = iou_summary.reset_index()
            iou_summary_reset["pipeline"] = pipeline_name
            all_iou.append(iou_summary_reset)

            # Calculate the IoU and Dice score
            mean_iou = iou_df["iou"].mean(skipna=True)
            mean_dice = iou_df["dice"].mean(skipna=True)
            macro_iou = iou_summary["mean_iou"].mean()
            macro_dice = iou_summary["mean_dice"].mean()

            macro_segmentation.append({
                "pipeline": pipeline_name,
                "mIoU": round(mean_iou, 4),
                "mDice": round(mean_dice, 4),
                "macro_iou": round(macro_iou, 4),
                "macro_dice": round(macro_dice, 4)
            })

            # Obtain the F1 score for presence metrics
            f1_df, macro_f1 = self.calculate_presence_f1(iou_df)
            f1_df["pipeline"] = pipeline_name
            f1_df.to_csv(f"{self.eval_config.OUTPUT_DIR}detection_f1_{pipeline_name}.csv", index=False)
            all_detection_f1.append(f1_df)
            macro_f1_pipeline.append({"pipeline": pipeline_name, "macro_f1": macro_f1})

            # Obtains the F1-score at the threshold of 0.50
            iou50_f1_df, macro_iou50_f1 = self.calculate_iou50_f1(iou_df)
            iou50_f1_df["pipeline"] = pipeline_name
            iou50_f1_df.to_csv(f"{self.eval_config.OUTPUT_DIR}iou50_f1_{pipeline_name}.csv", index=False)
            all_iou50_f1.append(iou50_f1_df)

        # Combines and stores all the results in separate consolidated files
        combined_iou = pd.concat(all_iou, ignore_index=True)
        combined_iou.to_csv(f"{self.eval_config.OUTPUT_DIR}class_iou_summary_all_pipelines.csv", index=False)

        combined_f1 = pd.concat(all_detection_f1, ignore_index=True)
        combined_f1.to_csv(f"{self.eval_config.OUTPUT_DIR}detection_f1_all_pipelines.csv", index=False)

        macro_f1_df = pd.DataFrame(macro_f1_pipeline)
        macro_f1_df.to_csv(f"{self.eval_config.OUTPUT_DIR}macro_f1.csv", index=False)

        combined_iou50_f1 = pd.concat(all_iou50_f1, ignore_index=True)
        combined_iou50_f1.to_csv(f"{self.eval_config.OUTPUT_DIR}iou50_f1_all_pipelines.csv", index=False)

        macro_seg_df = pd.DataFrame(macro_segmentation)
        macro_seg_df.to_csv(f"{self.eval_config.OUTPUT_DIR}segmentation_metrics.csv", index=False)

        final_results = (
            combined_f1
            .merge(combined_iou50_f1, on=["pipeline", "class"], how="outer")
            .merge(combined_iou, on=["pipeline", "class"], how="outer")
        )
        final_results.to_csv(f"{self.eval_config.OUTPUT_DIR}final_results_all_pipeline.csv", index=False)

        return combined_iou, combined_f1, combined_iou50_f1, macro_f1_df, macro_seg_df
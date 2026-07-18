import cv2
import numpy as np
import pandas as pd
import json
import os
from utils.log_config import get_logger
from utils.evaluation_config import EvaluatePipeline

logger = get_logger(__name__)

eval_config = EvaluatePipeline()
def extract_coco_box():
    """
    This method is used for reading the coco box and merge the multiple boxes into single box for all coarse items except for shoes. This is done because Grounding Dino gives one big box for object and not small multiple boxes
    """

    # Read the JSON from Anylabeling Output
    with open(eval_config.COCO_JSON_PATH, 'r') as gt_file:
        data = json.load(gt_file)

    category_mapping = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    gt_db = {}
    annotations_list = {}
    # Prepare annotation list
    for annotation in data.get("annotations",[]):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        category_name = category_mapping.get(category_id, str(category_id))
        x, y, w, h = annotation['bbox']
        box = [float(x), float(y), float(x + w), float(y + h)]
        if image_id not in annotations_list:
            annotations_list[image_id] = {}
        if category_name not in annotations_list[image_id]:
            annotations_list[image_id][category_name] = []
        annotations_list[image_id][category_name].append(box)
        

    # For each annotation merge the boxes to get a single box for Grounding Dino
    for image_id, categories in annotations_list.items():
        gt_db[image_id] = {}

        for cat_name, boxes in categories.items():
            if cat_name in ["top", "outer_top", "pants", "skirt", "dress"]:
                xmin = min(b[0] for b in boxes)
                ymin = min(b[1] for b in boxes)
                xmax = max(b[2] for b in boxes)
                ymax = max(b[3] for b in boxes)
                gt_db[image_id][cat_name] = [[xmin, ymin, xmax, ymax]]
                
            else:
                gt_db[image_id][cat_name] = boxes
    image_id_to_fn_dict = {img["id"]: img["file_name"] for img in data.get("images",[])}
    return gt_db, image_id_to_fn_dict

def extract_coco_box_for_yolo():
    """
    This method is used for reading the coco box same as extract_coco_box but this does not merge the boxes as this is for YOLO closed set object detection model that outputs multiple boxes and we are using SAM with logical OR to merge the boxes to SAM segmentation
    """
    with open(eval_config.COCO_JSON_PATH, 'r') as gt_file:
        data = json.load(gt_file)

    category_mapping = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    gt_db = {}
    annotations_list = {}
    for annotation in data.get("annotations",[]):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        category_name = category_mapping.get(category_id, str(category_id))
        x, y, w, h = annotation['bbox']
        box = [float(x), float(y), float(x + w), float(y + h)]
        if image_id not in annotations_list:
            annotations_list[image_id] = {}
        if category_name not in annotations_list[image_id]:
            annotations_list[image_id][category_name] = []
        annotations_list[image_id][category_name].append(box)
        

    for image_id, categories in annotations_list.items():
        gt_db[image_id] = {}

        for cat_name, boxes in categories.items():
            if cat_name == "bags":
                cat_name = "bag_wallet"
            gt_db[image_id][cat_name] = boxes
    image_id_to_fn_dict = {img["id"]: img["file_name"] for img in data.get("images",[])}
    return gt_db, image_id_to_fn_dict


def calculate_iou(true_box, prediction_box):
    """
    This method is used to calculate the IoU between the two boxes. takes boxes as input and then calculate the total merge area and union area to find the IoU
    """
    xA = max(true_box[0], prediction_box[0])
    yA = max(true_box[1], prediction_box[1])
    xB = min(true_box[2], prediction_box[2])
    yB = min(true_box[3], prediction_box[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    gt_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    pred_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])

    union = gt_area + pred_area - inter_area
    iou = float(inter_area / union) if union > 0 else 0.0

    return iou

def calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.50):
    """
    This is used to calculate the metrics required to calculate F1-score later. 
    The threshold of 0.50 to take the TP when matching items are found
    """
    if not pred_boxes and gt_boxes:
        return 0, 0, len(gt_boxes), []
    if pred_boxes and not gt_boxes:
        return 0, len(pred_boxes), 0, []
    
    matched_gt = set()
    TP, FP = 0, 0
    matched_ious = []

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt = -1
        for index, gt_box in enumerate(gt_boxes):
            if index in matched_gt:
                continue
            iou = calculate_iou(true_box=gt_box, prediction_box=pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = index
            
        if best_iou >= iou_threshold:
            TP +=1
            matched_gt.add(best_gt)
            matched_ious.append(best_iou)
        else:
            FP +=1
    
    FN = len(gt_boxes) - len(matched_gt)
    return TP, FP, FN, matched_ious

def evaluate(dino_model, images_dir, box_threshold, text_threshold,iou_threshold, area_threshold):
    """
    This method is used to evaluate the Grounding DINO boxes quality. Uses same code as that used for segmentation but cut off before segmentation to know the intermediate bounding box quality. This method was also used for threshold tuning to find the box and text threshold impact on the bounding boxes
    """
    gt_data, image_mapping = extract_coco_box()
    result_list = []
    count = 0

    for image_id, filename in image_mapping.items():
        if count % 10 == 0:
          print(f"done {count} images")
        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            print(f"{image_path}image not found")
            continue

        image_obj = cv2.imread(image_path)
        image_obj = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
        
        formatted_preds = dino_model.get_evaluate_boxes(
            image_obj=image_obj, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold,
            iou_threshold=iou_threshold, 
            area_threshold=area_threshold
        )
        image_gt = gt_data.get(image_id, {})
        for category in eval_config.TARGET_CLASS:
            cat_preds = formatted_preds.get(category, [])
            cat_gt = image_gt.get(category, [])
            
            TP, FP, FN, ious = calculate_metrics(cat_preds, cat_gt, iou_threshold=0.50)
            
            result_list.append({
                "category": category,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "IoU_Sum": sum(ious),
                "IoU_Count": len(ious)
            })
        # break
    df = pd.DataFrame(result_list)
    print(df)
    summary = df.groupby('category').sum().reset_index()
    summary['Precision'] = summary['TP'] / (summary['TP'] + summary['FP']).replace(0, np.nan)
    summary['Recall'] = summary['TP'] / (summary['TP'] + summary['FN']).replace(0, np.nan)
    summary['F1_Score'] = 2 * (summary['Precision'] * summary['Recall']) / (summary['Precision'] + summary['Recall'])
    summary['Mean_Box_IoU'] = summary['IoU_Sum'] / summary['IoU_Count'].replace(0, np.nan)

    summary = summary[['category', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score', 'Mean_Box_IoU']].fillna(0)
    return summary

def evaluate_yolo(yolo_model, images_dir, bag_threshold, box_threshold):
    """
    This method is used to evaluate the YOLO boxes quality. Same code as above just that this takes multiple boxes and not merge them as that is done for Grounding Dino
    """
    gt_data, image_mapping = extract_coco_box_for_yolo()
    result_list = []
    count = 0

    for image_id, filename in image_mapping.items():
        if count % 10 == 0 and count > 0:
          logger.info(f"done {count} images")
          # break
        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            logger.info(f"{image_path}image not found")
            continue

        image_obj = cv2.imread(image_path)
        image_obj = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
        
        formatted_preds = yolo_model.get_evaluation_boxes(
            image_obj=image_obj, 
            box_threshold=box_threshold, 
            bag_threshold=bag_threshold
        )
        if "bag_wallet" in formatted_preds:
            formatted_preds["bag"] = formatted_preds.pop("bag_wallet")
        image_gt = gt_data.get(image_id, {})
        for category in eval_config.TARGET_CLASS:
            cat_preds = formatted_preds.get(category, [])
            cat_gt = image_gt.get(category, [])
            
            TP, FP, FN, ious = calculate_metrics(cat_preds, cat_gt, iou_threshold=0.50)
            
            result_list.append({
                "category": category,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "IoU_Sum": sum(ious),
                "IoU_Count": len(ious)
            })
        count +=1
        # break
    df = pd.DataFrame(result_list)
    print(df)
    summary = df.groupby('category').sum().reset_index()
    summary['Precision'] = summary['TP'] / (summary['TP'] + summary['FP']).replace(0, np.nan)
    summary['Recall'] = summary['TP'] / (summary['TP'] + summary['FN']).replace(0, np.nan)
    summary['F1_Score'] = 2 * (summary['Precision'] * summary['Recall']) / (summary['Precision'] + summary['Recall'])
    summary['Mean_Box_IoU'] = summary['IoU_Sum'] / summary['IoU_Count'].replace(0, np.nan)

    summary = summary[['category', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score', 'Mean_Box_IoU']].fillna(0)
    return summary
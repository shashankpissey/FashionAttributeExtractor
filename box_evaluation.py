import torch
import pandas as pd

from config import GROUNDING_DINO_MODEL, OBJECT_DETECTOR_MODEL
from modules.DINO.DINO import GroundingDINO
from modules.objects_detector_separator import ObjectsDetectorAndSeparate
from evaluation.Evaluate_box_IoU import evaluate, evaluate_yolo

IMAGES_FOLDER = "/content/person_detect_output/"

def call_SAM():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groundedSAM = GroundingDINO(GROUNDING_DINO_MODEL, device)
    
    box_thresholds = [0.15, 0.20, 0.25, 0.30]
    text_thresholds = [0.20, 0.25, 0.30, 0.35]
    grid_search_results = []

    for b_threshold in box_thresholds:
        for t_threshold in text_thresholds:
            print(f"\nEvaluating -> Box Threshold: {b_threshold} | Text Threshold: {t_threshold}")
    
            metrics_table = evaluate(
                dino_model=groundedSAM, 
                images_dir=IMAGES_FOLDER, 
                box_threshold=b_threshold,
                text_threshold=t_threshold
            )
            metrics_table["box_threshold"] = b_threshold
            metrics_table["text_threshold"] = t_threshold
        
            print("Results")
            print(metrics_table)
            grid_search_results.append(metrics_table)
    final_metrics = pd.concat(grid_search_results, ignore_index=True)
    final_metrics.to_csv("box_iou_DINO.csv", index=False)


    iou_thresholds = [0.90, 0.85, 0.90]
    area_thresholds = [0.75, 0.85, 0.90]
    grid_search_results = []

    for b_threshold in iou_thresholds:
        for t_threshold in area_thresholds:
            print(f"\nEvaluating -> IOU Threshold: {b_threshold} | Area Threshold: {t_threshold}")
    
            metrics_table = evaluate(
                dino_model=groundedSAM, 
                images_dir=IMAGES_FOLDER, 
                box_threshold=0.2,
                text_threshold=0.25,
                iou_threshold=b_threshold,
                area_threshold=t_threshold
            )
            metrics_table["box_threshold"] = b_threshold
            metrics_table["text_threshold"] = t_threshold
        
            print("Results")
            print(metrics_table)
            grid_search_results.append(metrics_table)
    final_metrics = pd.concat(grid_search_results, ignore_index=True)
    final_metrics.to_csv("box_iou_area_DINO.csv", index=False)
    
    

def call_YOLO():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = GroundingDINO(ObjectsDetectorAndSeparate, device)
    
    detection_threshold = [0.15, 0.20, 0.30]
    box_threshold = [0.20, 0.30, 0.50]
    grid_search_results = []

    for d_threshold in detection_threshold:
        for b_threshold in box_threshold:
            print(f"\nEvaluating -> Detection Threshold: {d_threshold} | Box Threshold: {b_threshold}")
    
            metrics_table = evaluate_yolo(
                dino_model=yolo_model, 
                images_dir=IMAGES_FOLDER, 
                detection_threshold=detection_threshold,
                box_threshold=box_threshold
            )
            metrics_table["detection_threshold"] = d_threshold
            metrics_table["box_threshold"] = b_threshold
        
            print("Results")
            print(metrics_table)
            grid_search_results.append(metrics_table)
    final_metrics = pd.concat(grid_search_results, ignore_index=True)
    final_metrics.to_csv("box_iou_yolo.csv", index=False)
    
    
    
   



if __name__ == "__main__":

    call_SAM()
    call_YOLO()
    
    
    
    
    
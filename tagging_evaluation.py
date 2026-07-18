from evaluation.attribute_f1 import Evaluate_attribute_metrics
from config import GT_LOCATION, PIPELINE_B_SEG_IMAGES, PIPELINE_A_SEG_IMAGES, PIPELINE_C_SEG_IMAGES, PIPELINE_B_DIM_IMAGES, PIPELINE_B_FULL_DIM_IMAGES, PIPELINE_A_FULL_DIM_IMAGES, PIPELINE_C_FULL_DIM_IMAGES
if __name__ == "__main__":
    for data_folder in [PIPELINE_B_SEG_IMAGES, PIPELINE_A_SEG_IMAGES, PIPELINE_C_SEG_IMAGES, PIPELINE_B_DIM_IMAGES, PIPELINE_B_FULL_DIM_IMAGES,PIPELINE_A_FULL_DIM_IMAGES, PIPELINE_C_FULL_DIM_IMAGES]:
        file_name_to_save = data_folder.split("/")[-2]
        metric_evaluation = Evaluate_attribute_metrics(gt_file_path=GT_LOCATION, pred_file_path=data_folder+f"{file_name_to_save}_results.csv")
        metric_evaluation.evaluate(save_path=data_folder, segment_type=file_name_to_save)
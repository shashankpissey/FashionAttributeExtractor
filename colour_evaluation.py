import pandas as pd
from evaluation.colour_evaluation import EvaluateColour
from config import GT_LOCATION, PIPELINE_B_SEG_IMAGES
if __name__ == "__main__":
    for data_folder in [PIPELINE_B_SEG_IMAGES]:
        file_name_to_save = data_folder.split("/")[-2]
        result = {}
        for k in [3,4,5,6,7]:
            print(data_folder+f"{file_name_to_save}_{k}_results.csv")
            metric_evaluation = EvaluateColour(gt_path=GT_LOCATION, pred_path=data_folder+f"{file_name_to_save}_{k}_results.csv")
            acc = metric_evaluation.evaluate_primary_colour(pred_col="predicted_color", gt_col="tags.primary_color",save_path=data_folder, output_name=k)
            result[str(k)] = acc
        # print(result)
        pd.DataFrame(result).to_csv("all.csv")
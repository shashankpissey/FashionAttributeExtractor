from config import PIPELINE_B_SEG_IMAGES, GT_COLOUR_NAMES
from modules.Attribute_tagging.color_tagging import ColourTagger
from PIL import Image
import os
import pandas as pd
from utils.log_config import get_logger
from utils.timer import timer

logger = get_logger(__name__)

@timer
def colour_tagger():
    for data_folder in [PIPELINE_B_SEG_IMAGES]:
        file_name_to_save = data_folder.split("/")[-2]
        print(file_name_to_save)

        colour_tuning = ColourTagger(GT_COLOUR_NAMES)

        image_files = sorted([filename for filename in os.listdir(data_folder) if filename.lower().endswith(".png")])

        for k in [3,4,5,6,7]:
            # count = 0
            results = []
            for i, filename in enumerate(image_files):
                image = Image.open(os.path.join(data_folder, filename)).convert("RGB")
                result = colour_tuning.extract_colours(image, k)
                results.append({
                    "filename":filename,
                    "predicted_color": result.get("predicted_colour"),
                    "percentage": result.get("percentage"),
                    "delta_e": result.get("delta_e_distance")
                })
                # if count == 10:
                #     break
                # count += 1
            results_df = pd.DataFrame(results)
            output_path = data_folder+f"{file_name_to_save}_{k}_results.csv"
            results_df.to_csv(output_path, index=False)
            
            logger.info(f"Extraction complete! Saved {len(results)} rows to {output_path}")
    

if __name__ == "__main__":
    colour_tagger()
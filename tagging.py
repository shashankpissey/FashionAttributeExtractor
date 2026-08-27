from config import PIPELINE_B_SEG_IMAGES, PIPELINE_A_SEG_IMAGES, PIPELINE_C_SEG_IMAGES, PIPELINE_B_DIM_IMAGES, PIPELINE_B_FULL_DIM_IMAGES, PIPELINE_A_FULL_DIM_IMAGES, PIPELINE_C_FULL_DIM_IMAGES
from modules.Attribute_tagging.clip import CLIP_SigLip_Attribute_Extractor


def tag():
    for data_folder in [PIPELINE_B_SEG_IMAGES, PIPELINE_A_SEG_IMAGES, PIPELINE_C_SEG_IMAGES,PIPELINE_B_DIM_IMAGES, PIPELINE_B_FULL_DIM_IMAGES, PIPELINE_A_FULL_DIM_IMAGES, PIPELINE_C_FULL_DIM_IMAGES]:
            file_name_to_save = data_folder.split("/")[-2]
            print(file_name_to_save)
        
            extractor = CLIP_SigLip_Attribute_Extractor(device='cuda', model_name="SigLip", descriptive=True)
            
            results_df = extractor.extract_attributes(data_folder, batch_size=32, file_name_to_save=file_name_to_save)
            
            output_path = data_folder+f"{file_name_to_save}_results.csv"
            results_df.to_csv(output_path, index=False)
            
            print(f"Extraction complete! Saved {len(results_df)} rows to {output_path}")


if __name__ == "__main__":
    tag()
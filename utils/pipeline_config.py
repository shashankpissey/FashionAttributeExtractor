from enum import Enum

class PipelineConfig(Enum):
    PIPELINE_A = {
        "SEG_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_A/segmented_a",
        "FULL_DIM_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_A/full_dimmed/",
        "DIM_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_A/seg_dimmed/"
        }
    PIPELINE_B = {
        "SEG_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_B/segmented_b",
        "FULL_DIM_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_B/full_dimmed/",
        "DIM_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_B/seg_dimmed/"
        }
    PIPELINE_C = {
        "SEG_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_C/segmented_c",
        "FULL_DIM_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_C/full_dimmed/",
        "DIM_IMAGES": "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_C/seg_dimmed/"
        }

    @property
    def seg_dir(self):
        return self.value["SEG_IMAGES"]
    
    @property
    def dim_dir(self):
        return self.value["DIM_IMAGES"]
    
    @property
    def full_dim_dir(self):
        return self.value["FULL_DIM_IMAGES"]
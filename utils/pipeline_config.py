from enum import Enum
from config import PIPELINE_A_FULL_DIM_IMAGES,PIPELINE_A_SEG_IMAGES,PIPELINE_A_EVAL_IMAGES,PIPELINE_B_DIM_IMAGES,PIPELINE_B_FULL_DIM_IMAGES,PIPELINE_B_SEG_IMAGES,PIPELINE_B_EVAL_IMAGES,PIPELINE_C_FULL_DIM_IMAGES,PIPELINE_C_SEG_IMAGES,PIPELINE_C_EVAL_IMAGES

class PipelineConfig(Enum):
    PIPELINE_A = {
        "SEG_IMAGES": PIPELINE_A_SEG_IMAGES,
        "FULL_DIM_IMAGES": PIPELINE_A_FULL_DIM_IMAGES,
        "DIM_IMAGES": "",
        "EVAL_DIR": PIPELINE_A_EVAL_IMAGES
        }
    PIPELINE_B = {
        "SEG_IMAGES": PIPELINE_B_SEG_IMAGES,
        "FULL_DIM_IMAGES": PIPELINE_B_FULL_DIM_IMAGES,
        "DIM_IMAGES": PIPELINE_B_DIM_IMAGES,
        "EVAL_DIR": PIPELINE_B_EVAL_IMAGES
        }
    PIPELINE_C = {
        "SEG_IMAGES": PIPELINE_C_SEG_IMAGES,
        "FULL_DIM_IMAGES": PIPELINE_C_FULL_DIM_IMAGES,
        "DIM_IMAGES": "",
        "EVAL_DIR": PIPELINE_C_EVAL_IMAGES
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
    
    @property
    def eval_dir(self):
        return self.value["EVAL_DIR"]
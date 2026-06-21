DATASET_PATH = ""

# Person Object Detection
CLASS_ID = 0


# Models Used
PERSON_DETECTION_MODEL = "yolo26n.pt"
SEGMENTER_MODEL = "sayeed99/segformer-b2-fashion"
SEGMENTATION_MODEL = "mattmdjaga/segformer_b2_clothes"
# SEGMENTATION_MODEL = "matei-dorian/segformer-b5-finetuned-human-parsing"
OBJECT_DETECTOR_MODEL = "valentinafevu/yolos-fashionpedia"
SAM_MODEL = "facebook/sam-vit-base"
GROUNDING_DINO_MODEL ="IDEA-Research/grounding-dino-base"

DINO_PROMPT = "tshirt . offshoulder top . tank top . blouse . sweater . dress . jacket . coat . skirt . pants . bag . shoes . sunglasses . belt . hat ."
MUTUALLY_EXCLUSIVE_GARMENTS = [
            ("skirt", "pants"),
            ("skirt", "trousers"),
            ("trousers", "pants"),
            ("skirt", "dress"),
            ("outer_top", "top")
            ]


# mattmdjafa key


# # Input Location
# INPUT_IMAGES = ""
# # Save Locations
# CROPPED_IMAGES = ""
# PIPELINE_B_SEG_IMAGES = ""
# PIPELINE_B_FULL_DIM_IMAGES = ""
# PIPELINE_B_DIM_IMAGES = ""
# PIPELINE_A_SEG_IMAGES = ""
# PIPELINE_A_FULL_DIM_IMAGES = ""
# PIPELINE_C_SEG_IMAGES = ""
# PIPELINE_C_FULL_DIM_IMAGES = ""


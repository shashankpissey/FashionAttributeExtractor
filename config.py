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


# Evaluation 

## IoU
TARGET_CLASSES = ["bag", "top", "outer_top", "pants", "shoes", "skirt", "dress"]

GT_MAPPING = {
    "bag": ["bag"],
    "top": ["top","top#"],
    "outer_top": ["outer_top"],
    "pants": ["pants", "shorts", "spants"],  
    "shoes": ["shoes", "sho#"],
    "skirt": ["skirt"],
    "dress": ["dress"]
}

MASK_FILE_NAMES = {
    "bag": "bag_wallet",
    "top": "top",
    "outer_top": "outer_top",
    "pants": "pants",
    "shoes": "shoes",
    "skirt": "skirt",
    "dress": "dress"
}

COCO_JSON_PATH = "D:/Dissertation/Final/annotations_last.json"

# COCO_JSON_PATH = "D:/Dissertation/Test/Data/manual_tagging/Annotation_masks_details.json"


PIPELINES = ["Pipeline_A (Yolos)", "Pipeline_B (Segformer Hybrid)", "Pipeline_C (Grounding DINO)"]
# PIPELINES = ["Pipeline_B (Segformer Hybrid)"]
PRED_MASK_PATH = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/"

CLIPPREDICTION_CSV = []

METADATA_DIRECTORY = []

OUTPUT_DIR = "D:/Dissertation/Final/Data/Segmented_Files/results/"


# # Input Location
INPUT_IMAGES = ""
# Save Locations
CROPPED_IMAGES = ""
PIPELINE_B_SEG_IMAGES = ""
PIPELINE_B_FULL_DIM_IMAGES = ""
PIPELINE_B_EVAL_IMAGES = ""
PIPELINE_B_DIM_IMAGES = ""
PIPELINE_A_SEG_IMAGES = ""
PIPELINE_A_FULL_DIM_IMAGES = ""
PIPELINE_A_EVAL_IMAGES = ""
PIPELINE_C_SEG_IMAGES = ""
PIPELINE_C_FULL_DIM_IMAGES = ""
PIPELINE_C_EVAL_IMAGES = ""

# Input Location
INPUT_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/"
# Save Locations
CROPPED_IMAGES = "E:/Masters/Dissertation/Final/person_detect_output/"
PIPELINE_B_SEG_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/segmented_b/"
PIPELINE_B_FULL_DIM_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/full_dimmed/"
PIPELINE_B_DIM_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/seg_dimmed/"
PIPELINE_A_SEG_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_A/segmented_a/"
PIPELINE_A_FULL_DIM_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_A/full_dimmed/"
PIPELINE_C_SEG_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_C/segmented_c/"
PIPELINE_C_FULL_DIM_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_C/full_dimmed/"
PIPELINE_A_EVAL_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_A/eval_images"
PIPELINE_B_EVAL_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/eval_images"
PIPELINE_C_EVAL_IMAGES = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_C/eval_images"
PIPELINE_A_EVAL_IMAGES_CONF = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_A/eval_images/conf"
PIPELINE_B_EVAL_IMAGES_CONF = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/eval_images/conf"
PIPELINE_C_EVAL_IMAGES_CONF = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/Pipeline_C/eval_images/conf"

CLIP_PROMPT="E:/Masters/Dissertation/Code/Dissertation/FashionAttributeExtractor/data/prompt.json"
PROMPT_DESCRIPTION="E:/Masters/Dissertation/Code/Dissertation/FashionAttributeExtractor/data/prompt_description.json"
TAXONOMY_HIERARCHY="E:/Masters/Dissertation/Code/Dissertation/FashionAttributeExtractor/data/taxonomy_hierarchy_config.json"
COLOUR_MAPPING="E:/Masters/Dissertation/Code/Dissertation/FashionAttributeExtractor/data/colour_mapping.json"


GT_LOCATION = "D:/Dissertation/Test/Data/Segmented_Files/Validation/Baseline_f1.csv"


# Color

GT_COLOUR_NAMES = [
    'cream', 'gold', 'white', 'brown', 'beige', 'black', 'red', 'blue',
    'dark brown', 'light blue', 'grey', 'green', 'orange', 'light green',
    'peach', 'navy', 'dark blue', 'turquoise', 'tan', 'khaki',
    'light grey', 'burgundy', 'dark green', 'sage green', 'pink', 
    'lavender', 'silver', 'light brown', 'purple', 'maroon', 'lime',
    'olive green', 'light yellow', 'yellow', 'rust', 'dark grey', 'teal',
    'clear'
]
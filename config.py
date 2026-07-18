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

COCO_JSON_PATH = "D:/Dissertation/Test/Data/manual_tagging/Annotation_masks_details.json"

# PIPELINES = ["Pipeline_A (Closed Class Obj Detector + SAM)", "Pipeline_B (Segformer)", "Pipeline_C (Grounded SAM)"]

PIPELINES = ["Pipeline_A (Yolos)", "Pipeline_B (Segformer)", "Pipeline_C (Grounding DINO)"]

PRED_MASK_PATH = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/"

CLIPPREDICTION_CSV = []

METADATA_DIRECTORY = []

OUTPUT_DIR = "D:/Dissertation/Test/Data/Validation/results/"



# mattmdjafa key


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
INPUT_IMAGES = "E:/Masters/Dissertation/Code/backup/data/"
# Save Locations
CROPPED_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/person_detect_output/"
PIPELINE_B_SEG_IMAGES = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/segmented_b/"
PIPELINE_B_FULL_DIM_IMAGES = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/full_dimmed/"
PIPELINE_B_DIM_IMAGES = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/Pipeline_B/seg_dimmed/"
PIPELINE_A_SEG_IMAGES = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/Pipeline_A/segmented_a/"
PIPELINE_A_FULL_DIM_IMAGES = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/Pipeline_A/full_dimmed/"
PIPELINE_C_SEG_IMAGES = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/Pipeline_C/segmented_c/"
PIPELINE_C_FULL_DIM_IMAGES = "D:/Dissertation/Test/Data/Segmented_Files/Final_Segmented_Files/Pipeline_C/full_dimmed/"

CLIP_PROMPT="E:/Masters/Dissertation/Code/Dissertation/FashionAttributeExtractor/data/prompt.json"
PROMPT_DESCRIPTION="E:/Masters/Dissertation/Code/Dissertation/FashionAttributeExtractor/data/prompt_description.json"
TAXONOMY_HIERARCHY="E:/Masters/Dissertation/Code/Dissertation/FashionAttributeExtractor/data/taxonomy_hierarchy_config.json"


GT_LOCATION = "D:/Dissertation/Test/Data/Segmented_Files/Validation/Baseline_final.csv"


# Color

# GT_COLOR_NAMES = [
#     'cream', 'gold', 'white', 'brown', 'beige', 'black', 'red', 'blue',
#     'dark brown', 'light blue', 'grey', 'green', 'orange', 'light green',
#     'peach', 'navy', 'dark blue', 'turquoise', 'tan', 'khaki',
#     'light grey', 'burgundy', 'dark green', 'sage green', 'pink', 'olive',
#     'lavender', 'silver', 'light brown', 'purple', 'maroon', 'lime',
#     'olive green', 'light yellow', 'yellow', 'rust', 'dark grey', 'teal',
#     'clear'
# ]

GT_COLOR_NAMES = [
        'white', 'black', 'grey', 'silver',
        'blue', 'navy', 'turquoise',
        'red', 'burgundy',
        'green', 'khaki',
        'yellow', 'gold', 'orange',
        'pink', 'peach', 'purple',
        'brown', 'tan', 'beige', 'cream'
    ]

PARENT_COLOR_MAP = { 
    'light blue':'blue', 'dark blue':'blue', 'navy':'blue','turquoise':'blue', 'teal':'blue', 'blue': 'blue', 'green':'green', 'light green':'green', 'dark green':'green', 'sage green': 'green', 'olive':'green', 'olive green':'green', 'lime':'green', 'khaki':'green', 'red': 'red', 'burgundy':'red', 'maroon': 'red', 'rust': 'red', 'pink':'pink', 'peach':'pink', 'lavender':'purple', 'purple':'purple', 'orange': 'orange', 'gold':'yellow', 'yellow':'yellow', 'light yellow': 'yellow', 'brown':'brown','dark brown': 'brown', 'light brown':'brown', 'tan':'brown', 'beige':'beige','cream':'beige', 'black':'black', 'white':'white','clear':'white', 'grey':'grey','light grey': 'grey', 'dark grey': 'grey','silver':'grey'
}
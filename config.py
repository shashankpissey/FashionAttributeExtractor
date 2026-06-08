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


# mattmdjafa key


# Input Location
INPUT_IMAGES = "E:/Masters/Dissertation/Code/backup/data/"
# Save Locations
CROPPED_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/person_detect_output/"
PIPELINE_B_SEG_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_B/segmented_b/"
PIPELINE_B_FULL_DIM_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_B/full_dimmed/"
PIPELINE_B_DIM_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_B/seg_dimmed/"
PIPELINE_A_SEG_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_A/segmented_a"
PIPELINE_A_FULL_DIM_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_A/full_dimmed/"
PIPELINE_C_SEG_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_C/segmented_c"
PIPELINE_C_FULL_DIM_IMAGES = "E:/Masters/Dissertation/Code/backup/data/test/seg/pipeline_C/full_dimmed/"
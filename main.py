import os
import sys
import torch

from config import PERSON_DETECTION_MODEL, OBJECT_DETECTOR_MODEL, SAM_MODEL, INPUT_IMAGES, CROPPED_IMAGES, GROUNDING_DINO_MODEL

from modules.person_detector import PersonDetector
from modules.objects_detector_separator import ObjectsDetectorAndSeparate
from modules.DINO.DINO import GroundingDINO
import modules.Segformer.segformer_factory as sf
from utils.timer import timer
from utils.log_config import get_logger
from utils.create_folders import create_folders
from tagging import tag


logger = get_logger(__name__)

def extraction_driver(person_detector, segmenter_obj, separator, zero_shot_model, save=False, pipeline_name="PIPELINE_B"):
    """
        This method is the main method that is used control the entire flow of the program
        
    """

    try:
        
        for file_name in os.listdir(INPUT_IMAGES):
            if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', 'webp')):
                basename = os.path.splitext(file_name)[0]
                input_image_path = INPUT_IMAGES+file_name

                # print("Detecting person")

                # person_detect = person_detector.detect_person(input_image_path)

                # print("Detected person")

                # if person_detect:
                #     print(f"Person detected and segmentatin for person is being performed")
                #     cropped_person = person_detector.crop_person(input_image_path, person_detect)
                #     print("Segmentation Started")
                #     # segmenter_obj = Segmenter(SEGMENTATION_MODEL, device)
                #     masks_list = segmenter_obj.extract_segmentation(cropped_person, separator, save=save, basename=basename)
                #     print("Completed Main Segmentor and starting rest")
                #     if save:
                #         person_detector.save_person(cropped_person, output_dir)
                #         print("Save person")

                # else:
                #     print(f"No Person detected in the image {input_image_path}")

                # Multiple people

                all_people_box = person_detector.detect_all_people(input_image_path)
                logger.info(f"for image {basename}, detected all people {len(all_people_box)}")

                for person_no, box in enumerate(all_people_box):
                    person_basename = f"{basename}_p{person_no}"
                    expanded_crop, expanded_box = person_detector.crop_person_ex(input_image_path, box, expand_perc=0.45)
                    if expanded_crop is not None:
                        if pipeline_name.upper() == "PIPELINE_A":
                            mask_list = separator.seg_only_obj_det(expanded_crop, "",save, basename=person_basename,pipeline_name = pipeline_name.upper())
                        elif pipeline_name.upper() == "PIPELINE_B":
                            print("Segmentation Started")
                            mask_list = segmenter_obj.extract_seg(expanded_crop, separator, save=save, basename=person_basename, pipeline_name = pipeline_name.upper())
                        elif pipeline_name.upper() == "PIPELINE_C":
                            print("Segmentation Started")
                            mask_list = zero_shot_model.execute_dino_pipeline(expanded_crop, separator, save=save, basename=person_basename,pipeline_name=pipeline_name.upper())
                        else:
                            logger.error(f"Pipeline is incorrect {pipeline_name}")
                        if save:
                            person_detector.save_person(expanded_crop, CROPPED_IMAGES, person_basename)
            # break
    except Exception as e:
        print("Initialisation failed")
        print(e)
        sys.exit(1)

@timer
def execute_pipeline(pipeline_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("--------Start----------")
    person_detector = PersonDetector(PERSON_DETECTION_MODEL, device)
    stage = "LOOK"
    segmenter_obj = sf.segformer_factory(stage, device)
    # segmenter_obj = Segmenter(SEGMENTATION_MODEL, device)
    separator = ObjectsDetectorAndSeparate(OBJECT_DETECTOR_MODEL, device)
    groundedSAM = GroundingDINO(GROUNDING_DINO_MODEL, device)
    extraction_driver(person_detector=person_detector, segmenter_obj=segmenter_obj, separator=separator,zero_shot_model=groundedSAM, save=True, pipeline_name=pipeline_name)
    tag()

if __name__ == "__main__":
    # for pipeline in ["PIPELINE_A", "PIPELINE_C"]:
    create_folders()
    for pipeline in ["PIPELINE_A"]:
        execute_pipeline(pipeline_name=pipeline)
        
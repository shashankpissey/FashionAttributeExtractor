import json
import os
import numpy as np
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image

from utils.pipeline_config import PipelineConfig
from utils.log_config import get_logger
from utils.calculate_conf import calculate_conf_metadata
from ..objects_detector_separator import ObjectsDetectorAndSeparate
from config import DINO_PROMPT, MUTUALLY_EXCLUSIVE_GARMENTS

logger = get_logger(__name__)

class GroundingDINO:

    def __init__(self, model, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model).to(self.device)
        self.box_threshold = 0.2
        self.text_threshold = 0.25

    def get_obj_boxes(self, image_obj, text_prompt, box_threshold=0.2, text_threshold=0.25):
        """
        This method runs the inference on Grounding DINO based on the passed threshold and then returns the boxes and its correxponding labels. Code similar to huggingface for drawing the inference
        """
        image = Image.fromarray(image_obj)
        # Code similar to Hugging Face

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        print(results)
        prediction_dict = results[0]
        boxes = prediction_dict["boxes"].cpu().numpy().tolist()
        labels = prediction_dict["text_labels"]

        return boxes, labels
    
    def filter_items(self, boxes, category):
        """
        This method is used to filter the items based on the fashion garment domain to filter duplicate tops from dressed and bottoms that occupy whole image.
        """
        if not boxes:
            return None
        
        category = category.lower().strip()
        valid_boxes = []

        for box in boxes:
            if len(box) != 4:
                continue
            xmin, ymin, xmax, ymax = box
            box_height = ymax - ymin
            box_width = xmax - xmin
            ratio = box_height / box_width if box_width > 0 else 0

            if category in ["top", "tshirt"]:
                # 2.2 is derived from normal ratio from domain that a top norally has a width < 2 to its height
                if ratio > 2.2:
                    continue
            elif category in ["skirt", "pants"]:
                # If a pant or skirt is starting from the top and occupies most of the pixels then drop these
                # print(ymin)
                # print(box_height)
                if ymin < 50 and box_height > 1000:
                    continue
            valid_boxes.append(box)

        if not valid_boxes:
            valid_boxes = boxes

        if category in ["shoes", "earrings"]:
            sorted_boxes = sorted(valid_boxes, key=lambda area: (area[2] - area[0]) * (area[3] - area[1]), reverse=True)
            return sorted_boxes[:2]
        else:
            sorted_boxes = sorted(valid_boxes, key=lambda area: (area[2] - area[0]) * (area[3] - area[1]))
            return [sorted_boxes[0]]

    def calculate_iou(self, box1, box2):
        """
        Calculate the  IoU between boxes to find if it is a duplicate box
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area == 0:
            return 0.0
        box1area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = area / float(box1area + box2area - area)

        return iou

    def calculate_ioa(self, box1, box2):
        """
        Calculate the  IoA (Area) between boxes to find if it is a duplicate box
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        if box1area == 0:
            return 0.0

        ioa = area / float(box1area)


        return ioa
    
    def get_garment_conflict(self, box_dict, overlapping_area_threshold, overlapping_perc_threshold):
        """
        This method filters the overlapping boxes for mutually exclusive garments. Like Skirt and dress, Skirt and pants, dress and top.
        It takes the IoU and IoA to see if it is matching the threshold then it considers it duplicates and drops the duplicate box and keeps the first one 
        """
        box_dict_copy = box_dict.copy()
        keys = list(box_dict_copy.keys())
        if "dress" in keys:
            separates = ["top", "outer_top", "skirt", "pants", "trousers"]
            for item in separates:
                if item in box_dict_copy:
                    del box_dict_copy[item]
            keys = list(box_dict_copy.keys())
        
        keys = list(box_dict_copy.keys())
        # print(f"keys updated {keys}")
        
        for item1, item2 in MUTUALLY_EXCLUSIVE_GARMENTS:
            print(f"in loop {item1}, {item2}")
            if item1 in keys and item2 in keys:
                print("in this loop")
                box1 = self.filter_items(box_dict_copy[item1], item1)
                box2 = self.filter_items(box_dict_copy[item2], item2)

                if box1 and box2:
                    overlapping_perc = self.calculate_iou(box1[0], box2[0])
                    overlapping_area = self.calculate_ioa(box1[0], box2[0])
                    # print(item1, item2)
                    # print(overlapping_perc)
                    # print(overlapping_area)

                    if overlapping_perc > overlapping_perc_threshold and overlapping_area > overlapping_area_threshold:
                        del box_dict_copy[item2]
                        keys = list(box_dict_copy.keys())
        # print(box_dict_copy)

        return box_dict_copy
    
    def execute_dino_pipeline(self, image_obj, separator, save=False, basename="",pipeline_name="PIPELINE_C"):
        """
        This is used as main method to run full pipeline. 
        """

        current_pipeline = PipelineConfig[pipeline_name]
                
        boxes, labels = self.get_obj_boxes(image_obj=image_obj, text_prompt=DINO_PROMPT, box_threshold=self.box_threshold, text_threshold=self.text_threshold)

        discovered_boxes = {}
        EXPECTED_CLASSES = [c.strip() for c in DINO_PROMPT.split(".")]
        print(EXPECTED_CLASSES)

        for box, label in zip(boxes, labels):
            clean_label = label.lower().strip()
            if not clean_label == "":
                if clean_label not in EXPECTED_CLASSES:
                    matched_classes = [cat for cat in EXPECTED_CLASSES if cat in clean_label]
                    if matched_classes:
                        clean_label = matched_classes[0]
                    else:
                        continue
                if clean_label not in discovered_boxes:
                    discovered_boxes[clean_label] = []
                discovered_boxes[clean_label].append(box)


        normalized_boxes = {}
        for label, boxes in discovered_boxes.items():
            if label in ["tshirt", "tank top", "blouse", "sweater", "offshoulder top"]:
                key = "top"
            elif label in ["jacket", "coat"]:
                key = "outer_top"
            else:
                key = label
            
            if key not in normalized_boxes:
                normalized_boxes[key] = []
            normalized_boxes[key].extend(boxes)

        clean_boxes = self.get_garment_conflict(box_dict=normalized_boxes, overlapping_perc_threshold=0.9, overlapping_area_threshold=0.9)

        extracted_items = {}

        for category, box in clean_boxes.items():
            best_boxes = self.filter_items(boxes=box, category=category)

            if best_boxes:
                combined_mask = None
                for box in best_boxes:
                    mask = separator.sam_clipper(image=image_obj, box=box)
                    single_mask = (mask > 0).astype(np.uint8)
                    if combined_mask is None:
                        combined_mask = single_mask.astype(np.uint8)
                    else:
                        combined_mask = np.logical_or(combined_mask, single_mask).astype(np.uint8)

                h, w, c = image_obj.shape
                
                full_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                full_rgba[:, :, :3] = image_obj
                full_rgba[:, :, 3] = (combined_mask * 255).astype(np.uint8)
                
                masked = image_obj.copy()
                masked[combined_mask == 0] = 0
                print("In here")

                all_xmin = min(b[0] for b in best_boxes)
                all_ymin = min(b[1] for b in best_boxes)
                all_xmax = max(b[2] for b in best_boxes)
                all_ymax = max(b[3] for b in best_boxes)
                union_box = [float(all_xmin), float(all_ymin), float(all_xmax), float(all_ymax)]
                xmin, ymin, xmax, ymax = map(int, union_box)
                rgba_crop = full_rgba[ymin:ymax, xmin:xmax]
                final_crop_pil = Image.fromarray(rgba_crop, mode = "RGBA")
                eval_mask_pil = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")

                dimmed_rgb = (image_obj * 0.25).astype(np.uint8)
        
                bool_mask_3d = np.expand_dims(combined_mask > 0, axis=-1)
                full_dim_arr = np.where(bool_mask_3d, image_obj, dimmed_rgb)
                full_dim_pil = Image.fromarray(full_dim_arr)

                metadata = calculate_conf_metadata(
                    combined_mask,
                    union_box,
                    h,w
                )

                if category == "bag":
                    category = "bag_wallet"

                extracted_items[category] = {
                    "seg_crop": final_crop_pil,
                    "full_dim": full_dim_pil,
                    "eval_mask": eval_mask_pil,
                    "dino_box": best_boxes,
                    "metadata": metadata
                    }

                if save:
                    self.save_segments(img_pil=final_crop_pil, save_dir=current_pipeline.seg_dir, basename=basename, group_name=category)
                    self.save_segments(img_pil=full_dim_pil, save_dir=current_pipeline.full_dim_dir, basename=basename, group_name=category)
                    self.save_segments(img_pil=eval_mask_pil, save_dir=current_pipeline.eval_dir,basename=basename, group_name=category+"_mask")
                    bbox_data = {
                        "class_name": category,
                        "box_coordinates": extracted_items[category]["dino_box"],
                        "confidence": 1.0
                    }
                    with open(os.path.join(current_pipeline.eval_dir, f"{basename}_{category}_box.json"), "w") as f:
                        json.dump(bbox_data, f)
                    with open(os.path.join(current_pipeline.eval_dir+"/conf/", f"{basename}_{category}_meta.json"), "w") as f:
                        json.dump(metadata, f, indent=4)
        
        
        logger.info(f"For {basename} total {len(extracted_items)} with keys{extracted_items.keys()} are extracted")
        print(extracted_items)
        return extracted_items

    def save_segments(self, img_pil, save_dir, group_name, basename):
        save_path = os.path.join(save_dir, f"{basename}_{group_name}.png")

        img_pil.save(save_path)

    def get_evaluate_boxes(self, image_obj, box_threshold, text_threshold, iou_threshold, area_threshold):
        """
        This method same as the main method but without SAM. This output the boxes that is used to evaluate and tune the thresholds
        """
        boxes, labels = self.get_obj_boxes(image_obj=image_obj, text_prompt=DINO_PROMPT, box_threshold=box_threshold, text_threshold=text_threshold)

        discovered_boxes = {}
        EXPECTED_CLASSES = [c.strip() for c in DINO_PROMPT.split(".")]
        # print(EXPECTED_CLASSES)

        for box, label in zip(boxes, labels):
            clean_label = label.lower().strip()
            if not clean_label == "":
                if clean_label not in EXPECTED_CLASSES:
                    matched_classes = [cat for cat in EXPECTED_CLASSES if cat in clean_label]
                    if matched_classes:
                        clean_label = matched_classes[0]
                    else:
                        continue
                if clean_label not in discovered_boxes:
                    discovered_boxes[clean_label] = []
                discovered_boxes[clean_label].append(box)


        normalized_boxes = {}
        for label, boxes in discovered_boxes.items():
            if label in ["tshirt", "tank top", "blouse", "sweater", "offshoulder top"]:
                key = "top"
            elif label in ["jacket", "coat"]:
                key = "outer_top"
            else:
                key = label
            
            if key not in normalized_boxes:
                normalized_boxes[key] = []
            normalized_boxes[key].extend(boxes)

        clean_boxes = self.get_garment_conflict(box_dict=normalized_boxes, overlapping_perc_threshold=iou_threshold, overlapping_area_threshold=area_threshold)

        extracted_items = {}

        for category, box_list in clean_boxes.items():
            best_boxes = self.filter_items(boxes=box_list, category=category)
            if best_boxes:
                extracted_items[category] = best_boxes

        return extracted_items
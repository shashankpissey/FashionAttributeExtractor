import json
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torch
import os

from utils.pipeline_config import PipelineConfig
from .base_segmenter import BaseSegformer
from utils.log_config import get_logger

logger = get_logger(__name__)

class Segmenter(BaseSegformer):

    def __init__(self, model, device):
        
        super().__init__(model=model, device=device)

    def extract_seg(self, image_obj, separator, box, ex_box, conf_threshold=0.2, min_px=500, save=False, basename="", pipeline_name="PIPELINE_B"):
        try:
            min_px = int(min_px)
            conf_threshold = float(conf_threshold)
            current_pipeline = PipelineConfig[pipeline_name]
            image = Image.fromarray(image_obj)
            
            inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)

            with torch.inference_mode(), torch.autocast(device_type=self.device.type):
                outputs = self.seg_model(**inputs)
                logits = outputs.logits

                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=image.size[::-1],
                    mode="bilinear",
                    align_corners=False
                )
                probability = torch.softmax(upsampled_logits, dim=1)
                
                raw_segmentation = probability.argmax(dim=1)[0]
                confidence_map = probability.max(dim=1)[0][0]

            local_x1 = max(0, int(box[0] - ex_box[0]))
            local_y1 = max(0, int(box[1] - ex_box[1]))
            local_x2 = min(raw_segmentation.shape[1], int(box[2] - ex_box[0]))
            local_y2 = min(raw_segmentation.shape[0], int(box[3] - ex_box[1]))
            
            boundary_constraint = torch.zeros_like(raw_segmentation, dtype=torch.bool, device=self.device)
            boundary_constraint[local_y1:local_y2, local_x1:local_x2] = True
            
            predicted_segmentation = torch.where(boundary_constraint, raw_segmentation, torch.tensor(0, device=self.device))

            unique_classes, counts = torch.unique(predicted_segmentation, return_counts=True)
            class_dict = dict(zip(unique_classes.cpu().numpy(), counts.cpu().numpy()))
            
            upper_pixels = class_dict.get(4, 0)
            skirt_pixels = class_dict.get(5, 0)
            pants_pixels = class_dict.get(6, 0)
            dress_pixels = class_dict.get(7, 0)

            total_clothing_pixels = upper_pixels + skirt_pixels + pants_pixels + dress_pixels
            GROUPS = {}

            if total_clothing_pixels > min_px:
                dress_ratio = dress_pixels / total_clothing_pixels
                pants_ratio = pants_pixels / total_clothing_pixels
                upper_ratio = upper_pixels / total_clothing_pixels

                if dress_pixels > min_px and pants_pixels > min_px and dress_ratio > 0.15 and pants_ratio > 0.15:
                    GROUPS["dress"] = [4, 6, 7]
                    if skirt_pixels > min_px:
                        GROUPS["dress"].append(5)
                elif dress_pixels > min_px and dress_ratio > 0.20:
                    GROUPS = {"dress": [7]} 
                    if skirt_pixels > 0: 
                        GROUPS["dress"].append(5)    
                    if upper_pixels > 0 and upper_ratio < 0.25:
                        GROUPS["dress"].append(4)
                    else:
                        if upper_pixels > min_px:
                            GROUPS["upper"] = [4]
                    if pants_pixels > min_px and pants_ratio > 0.20: 
                        GROUPS["pants"] = [6]
                else:
                    if upper_pixels > min_px:
                        GROUPS["upper"] = [4]
                    if pants_pixels > min_px and skirt_pixels > min_px:
                        pants_bottom_ratio = pants_pixels / (pants_pixels + skirt_pixels)
                        if pants_bottom_ratio > 0.15: 
                            GROUPS["pants"] = [5, 6]
                        else:
                            GROUPS["skirt"] = [5, 6]
                    elif pants_pixels > min_px:
                        GROUPS["pants"] = [6]
                        
                    elif skirt_pixels > min_px:
                        GROUPS["skirt"] = [5]
                GROUPS["shoes"] = [9, 10]
                GROUPS["sunglasses"] = [3]

            raw_seg_cpu = raw_segmentation.cpu().numpy()
            face_mask = np.isin(raw_seg_cpu, [11,2,1,3])
            arms_mask = np.isin(raw_seg_cpu, [14, 15])
            legs_mask = np.isin(raw_seg_cpu, [12, 13])
            shoes_mask = np.isin(raw_seg_cpu, [9, 10])

            extracted_items = {}
            
            for group_name, class_ids in GROUPS.items():
                class_tensor = torch.tensor(class_ids, device=self.device)
                gpu_mask = torch.isin(predicted_segmentation, class_tensor) & (confidence_map > conf_threshold)
                
                mask = gpu_mask.cpu().numpy()
                min_pixels = 100 if group_name in ["shoes", "sunglasses"] else min_px

                if mask.sum() < min_pixels:
                    continue

                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue

                mask_uint8 = (mask * 255).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

                valid_blobs = 0
                clean_mask_uint8 = np.zeros_like(mask_uint8)
                for i in range(1, num_labels):
                    blob_area = stats[i, cv2.CC_STAT_AREA]
                    if blob_area > 300: 
                        valid_blobs += 1
                        clean_mask_uint8[labels == i] = 255
                if valid_blobs > 5:
                    print("In blob")
                    print(f"{basename}: The '{group_name}' is too highly fragmented ({valid_blobs} patches) hence rejected.")
                    continue
                
                coords = np.argwhere(clean_mask_uint8 > 0)
                if coords.size == 0:
                    continue

                h, w, c = image_obj.shape
                full_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                full_rgba[:, :, :3] = image_obj
                full_rgba[:, :, 3] = clean_mask_uint8
                full_size_pil = Image.fromarray(full_rgba)

                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                tight_crop_pil = Image.fromarray(full_rgba[y0:y1, x0:x1], mode="RGBA")

                if group_name in ["upper", "dress"]:
                    key = "top" if group_name == "upper" else "dress"
                    context = [face_mask, arms_mask] if group_name == "upper" else [face_mask, arms_mask, legs_mask, shoes_mask]
                    
                    upper_group = separator.detect_objects(image_obj, full_size_pil, save=False, basename=basename, group_name=group_name)
                    
                    if not upper_group or (len(upper_group) == 1 and key in upper_group):
                        full_dim = self.create_full_dim(image_obj=image_obj, target_mask=mask, opacity=0.25)
                        garment_dim = self.create_dim_external_mask(image_obj=image_obj, target_mask=mask, opacity=0.25, context_masks=context)
                        eval_mask_pil = Image.fromarray(mask_uint8, mode="L")
                        extracted_items[key] = {
                            "seg_crop": tight_crop_pil, 
                            "garment_dim": garment_dim, 
                            "full_dim": full_dim,
                            "eval_mask": eval_mask_pil}
                    else:
                        for layer_name, layer_data in upper_group.items():
                            if isinstance(layer_data, dict):
                                layer_pil = layer_data["image"]
                                yolo_box = layer_data.get("box")
                                yolo_score = layer_data.get("score")
                            else:
                                layer_pil = layer_data
                                yolo_box, yolo_score = None, None
                            layer_mask = np.array(layer_pil)[:, :, 3] > 0
                            layer_coords = np.argwhere(layer_mask)
                            if layer_coords.size > 0:
                                ly0, lx0 = layer_coords.min(axis=0)
                                ly1, lx1 = layer_coords.max(axis=0)
                                tight_layer_pil = Image.fromarray(np.array(layer_pil)[ly0:ly1, lx0:lx1], mode="RGBA")
                            else:
                                tight_layer_pil = layer_pil
                                
                            full_dim = self.create_full_dim(image_obj=image_obj, target_mask=layer_mask, opacity=0.25)
                            garment_dim = self.create_dim_external_mask(image_obj=image_obj, target_mask=layer_mask, opacity=0.25, context_masks=[face_mask, arms_mask])
                            eval_layer_mask_pil = Image.fromarray((layer_mask * 255).astype(np.uint8), mode="L")
                            extracted_items[layer_name] = {
                            "seg_crop": tight_layer_pil, 
                            "garment_dim": garment_dim, 
                            "full_dim": full_dim,
                            "eval_mask": eval_layer_mask_pil,
                            "yolo_box": yolo_box,
                            "yolo_score": yolo_score
                            }
                else:
                    context = []
                    if group_name in ["skirt", "pants"]:
                        context = [legs_mask, shoes_mask]
                    elif group_name == "shoes":
                        context = [legs_mask]
                    elif group_name == "sunglasses":
                        context = [face_mask]
                    elif group_name == "bag":
                        context = [face_mask, arms_mask]
                    
                    full_dim = self.create_full_dim(image_obj=image_obj, target_mask=mask, opacity=0.25)
                    garment_dim = self.create_dim_external_mask(image_obj=image_obj, target_mask=mask, opacity=0.25, context_masks=context)
                    eval_mask_pil = Image.fromarray(mask_uint8, mode="L")
                    extracted_items[group_name] = {
                        "seg_crop": tight_crop_pil,
                        "garment_dim": garment_dim,
                        "full_dim": full_dim,
                        "eval_mask": eval_mask_pil
                        }

                # print(extracted_items)
                if save:
                    for item_name, item_dict in extracted_items.items():
                        item_dict["seg_crop"].save(os.path.join(current_pipeline.seg_dir, f"{basename}_{item_name}.png"))
                        item_dict["garment_dim"].save(os.path.join(current_pipeline.dim_dir, f"{basename}_{item_name}.png"))
                        item_dict["full_dim"].save(os.path.join(current_pipeline.full_dim_dir, f"{basename}_{item_name}.png"))
                        item_dict["eval_mask"].save(os.path.join(current_pipeline.eval_dir, f"{basename}_{item_name}_mask.png"))
                        if item_dict.get("yolo_box") is not None:
                            bbox_data = {
                                "class_name": item_name,
                                "box_coordinates": item_dict["yolo_box"],
                                "confidence": item_dict["yolo_score"]
                            }
                            with open(os.path.join(current_pipeline.eval_dir, f"{basename}_{item_name}_box.json"), "w") as f:
                                json.dump(bbox_data, f)

        except Exception as e:
            print(f"Error in extract_seg: {e}")
    
        logger.info(f"For {basename} total {len(extracted_items)} with keys {list(extracted_items.keys())} are extracted")
        return extracted_items


    def save_segments(self, img_pil, save_dir, group_name, basename):
        save_path = os.path.join(save_dir, f"{basename}_{group_name}.png")
        img_pil.save(save_path)

    def create_dim_external_mask(self, image_obj, target_mask, context_masks=None, opacity=0.25):

        h, w = image_obj.shape[:2]
        alpha_mask = np.zeros((h,w), dtype=np.uint8)
        if context_masks:
            combined_context = np.logical_or.reduce(context_masks)
            alpha_mask[combined_context] = int(opacity * 255)

        alpha_mask[target_mask] = 255

        rgba_mask = np.zeros((h,w, 4), dtype=np.uint8)
        rgba_mask[:, :, :3] = image_obj
        rgba_mask[:, :, 3] = alpha_mask

        output = Image.new("RGB", (w,h), (0,0,0))
        output.paste(Image.fromarray(rgba_mask), (0,0), Image.fromarray(alpha_mask))

        return output

    def create_full_dim(self, image_obj, target_mask, opacity=0.25):
        h, w = image_obj.shape[:2]
        alpha_mask = np.full((h,w), int(opacity * 255), dtype=np.uint8)
        alpha_mask[target_mask] = 255
        rgba_mask = np.zeros((h,w, 4), dtype=np.uint8)
        rgba_mask[:, :, :3] = image_obj
        rgba_mask[:, :, 3] = alpha_mask
        output = Image.new("RGB", (w,h), (0,0,0))
        output.paste(Image.fromarray(rgba_mask), (0,0), Image.fromarray(alpha_mask))

        return output

    
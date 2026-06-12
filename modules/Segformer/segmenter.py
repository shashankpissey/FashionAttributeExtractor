import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torch
import os

from utils.pipeline_config import PipelineConfig
from .base_segmenter import BaseSegformer


class Segmenter(BaseSegformer):

    def __init__(self, model, device):
        
        super().__init__(model=model, device=device)

    def extract_seg(self, image_obj, separator, box, ex_box, conf_threshold=0.2, min_px=500, save=False, basename="", pipeline_name="PIPELINE_B"):

        try:
            min_px= int(min_px)
            conf_threshold = float(conf_threshold)
            current_pipeline = PipelineConfig[pipeline_name]
            image = Image.fromarray(image_obj)
            
            inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.seg_model(**inputs)

            logits = outputs.logits

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            probability = torch.softmax(upsampled_logits, dim=1)
            raw_segmentation = probability.argmax(dim=1)[0].numpy()
            
            face_mask = np.isin(raw_segmentation, [11,2,1,3])
            arms_mask = np.isin(raw_segmentation, [14, 15])
            legs_mask = np.isin(raw_segmentation, [12, 13])
            shoes_mask = np.isin(raw_segmentation, [9, 10])

            local_x1 = int(box[0] - ex_box[0])
            local_y1 = int(box[1] - ex_box[1])
            local_x2 = int(box[2] - ex_box[0])
            local_y2 = int(box[3] - ex_box[1])
            h, w = raw_segmentation.shape
            local_x1, local_y1 = max(0, local_x1), max(0, local_y1)
            local_x2, local_y2 = min(w, local_x2), min(h, local_y2)
            boundary_constraint = np.zeros_like(raw_segmentation, dtype=bool)
            boundary_constraint[local_y1:local_y2, local_x1:local_x2] = True
            
            predicted_segmentation = np.where(boundary_constraint, raw_segmentation, 0)

            unique_classes, counts = np.unique(predicted_segmentation, return_counts=True)
            print(unique_classes, counts)
            class_dict = dict(zip(unique_classes, counts))
            
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
                    if pants_pixels > min_px:
                        GROUPS["pants"] = [6]
                    if skirt_pixels > min_px:
                        if pants_pixels > 0:
                            skirt_pants_bool = skirt_pixels / (pants_pixels + skirt_pixels)
                            if skirt_pants_bool > 0.25:
                                GROUPS["skirt"] = [5]
                            else:
                                GROUPS["pants"].append(5)
                        else:
                            GROUPS["skirt"] = [5]
                GROUPS["shoes"] = [9, 10]
                GROUPS["sunglasses"] = [3]
            print("Groups Completed")
            confidence_map = probability.max(dim=1)[0][0].cpu().numpy()
            extracted_items = {}
            for group_name, class_ids in GROUPS.items():

                mask = np.isin(predicted_segmentation, class_ids)
                mask = mask & (confidence_map > conf_threshold)
                print(group_name)
                print(mask.sum())
                min_pixels = 100 if group_name in ["shoes", "sunglasses"] else min_px

                if mask.sum() < min_pixels:
                    continue

                full_mask = mask
                coords = np.argwhere(full_mask)
                if coords.size == 0:
                    continue

                mask_uint8 = (full_mask * 255).astype(np.uint8)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                
                valid_blobs = 0
                for i in range(1, num_labels):
                    blob_area = stats[i, cv2.CC_STAT_AREA]
                    if blob_area > 300: 
                        valid_blobs += 1
                if valid_blobs > 5:
                    print("In blob")
                    print(f"{basename}: The '{group_name}' is too highly fragmented ({valid_blobs} patches) hence rejected.")
                    continue

                h, w, c = image_obj.shape
                
                full_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                full_rgba[:, :, :3] = image_obj
                full_rgba[:, :, 3] = (full_mask * 255).astype(np.uint8)
                full_size_pil = Image.fromarray(full_rgba)

                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                tight_crop_pil = Image.fromarray(full_rgba[y0:y1, x0:x1], mode="RGBA")

                if group_name in ["upper", "dress"]:
                    print("in upper")
                    if group_name == "upper":
                        key = "top"
                        context = [face_mask, arms_mask]
                    else:
                        key = "dress"
                        context = [face_mask, arms_mask, legs_mask, shoes_mask]
                    
                    upper_group = separator.detect_objects(image_obj, full_size_pil, save=False, basename=basename, group_name=group_name)
                    if not upper_group or (len(upper_group) == 1 and key in upper_group):
                            print("no multilayer found")
                            full_dim = self.create_full_dim(image_obj=image_obj,target_mask=full_mask, opacity=0.25)
                            garment_dim = self.create_dim_external_mask(image_obj=image_obj,target_mask=full_mask, opacity=0.25, context_masks=context)

                            extracted_items[key] = {
                                "seg_crop": tight_crop_pil,
                                "garment_dim": garment_dim,
                                "full_dim": full_dim
                            }
                    else:
                        print("top or outer top found")
                        for layer_name, layer_pil in upper_group.items():
                            layer_mask = np.array(layer_pil)[:, :, 3] > 0
                            layer_coords = np.argwhere(layer_mask)
                            if layer_coords.size > 0:
                                ly0, lx0 = layer_coords.min(axis=0)
                                ly1, lx1 = layer_coords.max(axis=0)
                                layer_rgba = np.array(layer_pil)
                                tight_layer_pil = Image.fromarray(layer_rgba[ly0:ly1, lx0:lx1], mode = "RGBA")
                            else:
                                tight_layer_pil = layer_pil
                            full_dim = self.create_full_dim(image_obj=image_obj,target_mask=layer_mask, opacity=0.25)
                            garment_dim = self.create_dim_external_mask(image_obj=image_obj,target_mask=layer_mask, opacity=0.25, context_masks=[face_mask, arms_mask])

                            extracted_items[layer_name] = {
                                "seg_crop": tight_layer_pil,
                                "garment_dim": garment_dim,
                                "full_dim": full_dim
                            }
                else:
                    print(f"Here {group_name}")
                    context = []
                    if group_name in ["skirt", "pants"]:
                        context = [legs_mask, shoes_mask]
                    elif group_name == "shoes":
                        context = [legs_mask]
                    elif group_name == "sunglasses":
                        context = [face_mask]
                    elif group_name in ["bag"]:
                        context = [face_mask, arms_mask]
                    full_dim = self.create_full_dim(image_obj=image_obj,target_mask=full_mask, opacity=0.25)
                    garment_dim = self.create_dim_external_mask(image_obj=image_obj,target_mask=full_mask, opacity=0.25, context_masks=context)

                    extracted_items[group_name] = {
                                "seg_crop": tight_crop_pil,
                                "garment_dim": garment_dim,
                                "full_dim": full_dim
                            }
                if save:
                    for item_name, item_pil in extracted_items.items():
                        crop_path = os.path.join(current_pipeline.seg_dir, f"{basename}_{item_name}.png")
                        dim_path = os.path.join(current_pipeline.dim_dir, f"{basename}_{item_name}.png")
                        full_dim_path = os.path.join(current_pipeline.full_dim_dir, f"{basename}_{item_name}.png")

                        item_pil["seg_crop"].save(crop_path)
                        item_pil["garment_dim"].save(dim_path)
                        item_pil["full_dim"].save(full_dim_path)
                        
        except Exception as e:
            print(e)
        print(extracted_items)
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


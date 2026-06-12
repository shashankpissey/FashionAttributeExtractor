import cv2
import torch
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection, SamProcessor, SamModel
from PIL import Image
import os
from ultralytics import SAM

from utils.pipeline_config import PipelineConfig


class ObjectsDetectorAndSeparate:

    def __init__(self, obj_model, device):
        self.device = device
        self.obj_processor = YolosImageProcessor.from_pretrained(obj_model)
        self.obj_model = YolosForObjectDetection.from_pretrained(obj_model).to(device).eval()
        self.sam_model_obj = SAM("mobile_sam.pt")

    def sam_clipper(self, image, box):
        results = self.sam_model_obj(image, bboxes = box, verbose=False)
        mask_data = results[0].masks.data[0].cpu().numpy()
        best_output_mask = (mask_data * 255).astype(np.uint8) 

        return best_output_mask
    
    def detect_objects(self, image_obj, cropped_image, save, basename, group_name, min_pixels=2500):

        save_dir = "E:/Masters/Dissertation/Code/backup/data/test/seg/test1"
        obj_dict = {}
        
        try:
            obj_inputs = self.obj_processor(images=image_obj, return_tensors="pt").to(self.device)
            print("Obj Detection Started")

            with torch.no_grad():
                obj_outputs = self.obj_model(**obj_inputs)

            print("object Detection Ended")

            h, w = image_obj.shape[:2]
            target_sizes = torch.tensor([[h,w]])
            results = self.obj_processor.post_process_object_detection(outputs=obj_outputs, target_sizes=target_sizes, threshold=0.30)[0]
            print("obtained the results")
            # print(results)
        

            upper_classes = [0, 1]
            outer_classes = [2, 3, 4, 5, 9, 12]
            accessories_classes = [24, 15, 18, 25, 14, 16, 19]


            image_rgba = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2RGBA)
            detected_classes = [label.item() for label in results["labels"]]
            detected_perc = [label.item() for label in results["scores"]]
            print(detected_classes)
            print(detected_perc)
            has_coat = any (coat in outer_classes for coat in detected_classes)

            best_detections = {}
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                c_id = label.item()
                if c_id not in best_detections or score > best_detections[c_id]["score"]:
                    best_detections[c_id] = {"score": score, "label": label, "box": box}

            for det in best_detections.values():
                score = det["score"]
                label = det["label"]
                box = det["box"]
                
                class_id = label.item()
                box_coordinates = [float(i) for i in box.tolist()]
                class_name = self.obj_model.config.id2label[class_id].replace(", ","_").replace(" ","_")

                if class_id in accessories_classes:
                    mask = self.sam_clipper(image_obj, box_coordinates)

                    if np.count_nonzero(mask) > 500:

                        accessories_rgba = image_rgba.copy()
                        accessories_rgba[:, :, 3] = mask
                        accessory_image = Image.fromarray(accessories_rgba)

                        obj_dict[class_name] = accessory_image
                        if save:
                            self.save_segments(accessory_image, save_dir, class_name, basename)

                elif class_id in upper_classes and group_name == "upper":
                    mask = self.sam_clipper(image_obj, box_coordinates)

                    upper_array = np.array(cropped_image)
                    rgb = upper_array[:, :, :3]
                    segment_masks = upper_array[:, :, 3]

                    h1, w1 = segment_masks.shape
                    mask = cv2.resize(mask, (w1,h1), interpolation=cv2.INTER_NEAREST)

                    final_top = cv2.bitwise_and(segment_masks, mask)

                    if has_coat:
                        final_outer_top = cv2.bitwise_xor(segment_masks, final_top)
                        outer_pixels = np.count_nonzero(final_outer_top)
                        total_upper_pixels = np.count_nonzero(segment_masks)

                        outer_ratio = outer_pixels / total_upper_pixels if total_upper_pixels > 0 else 0

                        print(np.count_nonzero(final_outer_top))
                        
                        if np.count_nonzero(final_outer_top) > 1000 and outer_ratio > 0.20:
                            outer_top = Image.fromarray(np.dstack((rgb, final_outer_top)))
                            top = Image.fromarray(np.dstack((rgb, final_top)))
                            obj_dict["top"] = top
                            obj_dict["outer_top"] = outer_top
                            # del items_list["upper"]
                            if save:
                                self.save_segments(top, save_dir, "top", basename)
                                self.save_segments(outer_top, save_dir, "outer_top", basename)
                    
                        else:
                            print("Small coat area found hence ignored the layers and full image retained")
                            obj_dict["top"] = cropped_image
                    else:
                        print("No Coat so no separation")
                        obj_dict["top"] = cropped_image
                else:
                    print("No upper class found")
            
            if group_name == "dress":
                obj_dict["dress"] = cropped_image
            elif group_name == "upper" and "top" not in obj_dict:
                obj_dict["top"] = cropped_image

            return obj_dict
        except Exception as e:
            print(e)

    def save_segments(self, img_pil, save_dir, group_name, basename):
        save_path = os.path.join(save_dir, f"{basename}_{group_name}.png")
        img_pil.save(save_path)

    def calculate_iou(self, box1, box2):
            x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            if inter_area == 0: return 0
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return inter_area / (box1_area + box2_area - inter_area)

    
    def seg_only_obj_det(self, image_obj, save_dir, save, basename, pipeline_name = "Pipeline_B"):
        

        current_pipeline = PipelineConfig[pipeline_name]
        save_dir = current_pipeline.seg_dir
        obj_dict = {}

        try:
            obj_inputs = self.obj_processor(images=image_obj, return_tensors="pt").to(self.device)
            print("Obj Detection Started")

            with torch.no_grad():
                obj_outputs = self.obj_model(**obj_inputs)

            print("Object Detection Ended")

            h, w = image_obj.shape[:2]
            target_sizes = torch.tensor([[h, w]])
            results = self.obj_processor.post_process_object_detection(
                outputs=obj_outputs, target_sizes=target_sizes, threshold=0.20 
            )[0]

            upper_classes = [0, 1] 
            outer_classes = [2, 3, 4, 5, 9, 12]                   
            pants_class = 6
            skirt_class = 8
            dress_class = 10
            jumpsuit_class = 11
            shoes_class = 23                                            
            
            full_body_classes = [dress_class, jumpsuit_class]
            accessories_classes = [14, 15, 16, 18, 24, 25, 19, 13]
            allowed_classes = set(upper_classes + outer_classes + [pants_class, skirt_class, dress_class, jumpsuit_class, shoes_class] + accessories_classes + [2, 7])
            print(results)
            

            h, w = image_obj.shape[:2]
            total_area = h * w

            image_rgba = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2RGBA)

            valid_detections = [] 
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                custom_filter_threshold = 0.60 if label.item() in accessories_classes else 0.20
                if score.item() < custom_filter_threshold:
                    continue
                
                c_id = label.item()
                
                if c_id not in allowed_classes:
                    continue
                if c_id == 7:
                    c_id = pants_class

                box_coords = box.tolist()
                
                is_duplicate = False
                for i, accepted in enumerate(valid_detections):
                    if self.calculate_iou(box_coords, accepted["box"]) > 0.75:
                        is_duplicate = True
                        if score.item() > accepted["score"]:
                            valid_detections[i] = {"score": score.item(), "c_id": c_id, "box": box_coords}
                        break
                if not is_duplicate:
                    valid_detections.append({"score": score.item(), "c_id": c_id, "box": box_coords})

            grouped_detections = {}
            for det in valid_detections:
                c_id = det["c_id"]
                if c_id not in grouped_detections:
                    grouped_detections[c_id] = {"score": 0, "boxes": []}
                grouped_detections[c_id]["boxes"].append(det["box"])
                grouped_detections[c_id]["score"] = max(grouped_detections[c_id]["score"], det["score"])

            has_full_body = dress_class in grouped_detections or jumpsuit_class in grouped_detections
            if has_full_body:
                full_body_score = max(grouped_detections.get(dress_class, {}).get("score", 0), 
                                      grouped_detections.get(jumpsuit_class, {}).get("score", 0))
                
                top_score = max([grouped_detections.get(c, {}).get("score", 0) for c in upper_classes + [4]], default=0)
                bottom_score = max([grouped_detections.get(c, {}).get("score", 0) for c in [pants_class, skirt_class]], default=0)
                
                separates_score = max(top_score, bottom_score)

                if full_body_score > separates_score:
                    for c in upper_classes + [pants_class, skirt_class]:
                        grouped_detections.pop(c, None)
                else:
                    grouped_detections.pop(dress_class, None)
                    grouped_detections.pop(jumpsuit_class, None)

            class_masks = {}
            for c_id, data in grouped_detections.items():
                if c_id in upper_classes:
                    class_name = "top"
                elif c_id in outer_classes:
                    class_name = "outer_top"
                elif c_id == pants_class:
                    class_name = "pants"
                elif c_id == skirt_class:
                    class_name = "skirt"  
                elif c_id in full_body_classes:
                    class_name = "dress"
                elif c_id == shoes_class:
                    class_name = "shoes"
                elif c_id in accessories_classes:
                    class_name = self.obj_model.config.id2label[c_id].replace(", ", "_").replace(" ", "_")
                else:
                    class_name = f"unknown_{c_id}"
                
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                
                for box in data["boxes"]:
                    mask = self.sam_clipper(image_obj, box)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                
                class_masks[c_id] = {"name": class_name, "mask": combined_mask}

            garment_min_area = total_area * 0.001
            accessory_min_area = total_area * 0.0005

            for c_id, data in class_masks.items():
                class_name = data["name"]
                mask = data["mask"]

                if c_id in [pants_class, skirt_class]:
                    continue

                if (c_id in upper_classes or c_id in outer_classes or 
                    c_id == pants_class or c_id in full_body_classes or 
                    c_id in accessories_classes or c_id == shoes_class):
                    
                    threshold = accessory_min_area if (c_id in accessories_classes or c_id == shoes_class) else garment_min_area
                    
                    if np.count_nonzero(mask) > threshold:
                        processed_items = self.get_garment_crops(mask, image_rgba, image_obj)
                        obj_dict[class_name] = processed_items
                        if save:
                            crop_path = os.path.join(save_dir, f"{basename}_{class_name}_crop.png")
                            full_dim_path = os.path.join(current_pipeline.full_dim_dir, f"{basename}_{class_name}_dim.jpg")
                            
                            processed_items["seg_crop"].save(crop_path)
                            processed_items["full_dim"].convert("RGB").save(full_dim_path)

            if skirt_class in class_masks and pants_class in class_masks:
                print("Detected skirt worn over pants. Separating layers...")
                skirt_mask = class_masks[skirt_class]["mask"]
                pants_mask = class_masks[pants_class]["mask"]
                
                pure_pants_mask = cv2.bitwise_and(pants_mask, cv2.bitwise_not(skirt_mask))
                
                if np.count_nonzero(skirt_mask) > garment_min_area:
                    name = class_masks[skirt_class]["name"]
                    obj_dict[name] = self.get_garment_crops(skirt_mask, image_rgba, image_obj)
                    if save:
                        obj_dict[name]["seg_crop"].save(os.path.join(save_dir, f"{basename}_{name}_crop.png"))
                        obj_dict[name]["full_dim"].convert("RGB").save(os.path.join(current_pipeline.full_dim_dir, f"{basename}_{name}_dim.jpg"))
                    
                if np.count_nonzero(pure_pants_mask) > garment_min_area:
                    name = class_masks[pants_class]["name"]
                    obj_dict[name] = self.get_garment_crops(pure_pants_mask, image_rgba, image_obj)
                    if save:
                        obj_dict[name]["seg_crop"].save(os.path.join(save_dir, f"{basename}_{name}_crop.png"))
                        obj_dict[name]["full_dim"].convert("RGB").save(os.path.join(current_pipeline.full_dim_dir, f"{basename}_{name}_dim.jpg"))
            else:
                for target_class in [skirt_class, pants_class]:
                    if target_class in class_masks:
                        mask = class_masks[target_class]["mask"]
                        name = class_masks[target_class]["name"]
                        if np.count_nonzero(mask) > garment_min_area:
                            obj_dict[name] = self.get_garment_crops(mask, image_rgba, image_obj)
                            if save:
                                obj_dict[name]["seg_crop"].save(os.path.join(save_dir, f"{basename}_{name}.png"))
                                obj_dict[name]["full_dim"].convert("RGB").save(os.path.join(current_pipeline.full_dim_dir, f"{basename}_{name}.jpg"))

            print(obj_dict)

            return obj_dict

        except Exception as e:
            print(f"Error during segmentation: {e}")
            return obj_dict
        
    def get_garment_crops(self, mask, image_rgba, image_obj):
        y_indices, x_indices = np.nonzero(mask)
        y0, y1 = np.min(y_indices), np.max(y_indices)
        x0, x1 = np.min(x_indices), np.max(x_indices)
        
        tight_rgba = image_rgba.copy()
        tight_rgba[:, :, 3] = mask
        tight_crop_pil = Image.fromarray(tight_rgba[y0:y1+1, x0:x1+1], mode="RGBA")
        
        dimmed_rgb = (image_obj * 0.25).astype(np.uint8)
        
        bool_mask_3d = np.expand_dims(mask > 0, axis=-1)
        full_dim_arr = np.where(bool_mask_3d, image_obj, dimmed_rgb)
        full_dim_pil = Image.fromarray(full_dim_arr)
        
        return {
            "seg_crop": tight_crop_pil,
            "full_dim": full_dim_pil
        }
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torch
import os

from .base_segmenter import BaseSegformer


class SegmenterAll(BaseSegformer):

    def __init__(self, model, device):
        
        super().__init__(model=model, device=device)

    

    def save_segments(self, img_pil, save_dir, group_name, basename):
        save_path = os.path.join(save_dir, f"{basename}_{group_name}.png")
        img_pil.save(save_path)


    def extract_seg(self, image_obj, separator, box, ex_box, conf_threshold=0.2, min_px=500, save=False, basename=""):

        min_px= int(min_px)
        conf_threshold = float(conf_threshold)
        image = Image.fromarray(image_obj)
        # Code similar to Hugging Face
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
        print("\n--- Detected Clothes / Regions ---")
        for class_id, count in zip(unique_classes, counts):
            class_name = self.seg_model.config.id2label.get(class_id, f"Unknown_ID_{class_id}")
            print(f"Class: {class_name:<20} (ID: {class_id:<3}) -> Pixels: {count}")
        class_dict = dict(zip(unique_classes, counts))
        
        return "extracted_items"

    def create_dim_external_mask(self, image_obj, targer_mask, save, group_name, basename, context_masks=None, opacity=0.25):

        save_dir = "dimmed"
        h, w = image_obj.shape[:2]
        alpha_mask = np.zeros((h,w), dtype=np.uint8)
        if context_masks:
            combined_context = np.logical_or.reduce(context_masks)
            alpha_mask[combined_context] = int(opacity * 255)

        alpha_mask[targer_mask] = 255

        rgba_mask = np.zeros((h,w, 4), dtype=np.uint8)
        rgba_mask[:, :, :3] = image_obj
        rgba_mask[:, :, 3] = alpha_mask

        output = Image.new("RGB", (w,h), (0,0,0))
        output.paste(Image.fromarray(rgba_mask), (0,0), Image.fromarray(alpha_mask))

        if save:
            self.save_segments(output, save_dir, group_name, basename)

        return output

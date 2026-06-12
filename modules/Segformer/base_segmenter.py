from abc import ABC, abstractmethod
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class BaseSegformer(ABC):
    

    def __init__(self, model, device):
        self.device = device
        self.seg_processor = SegformerImageProcessor.from_pretrained(model)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(model).to(device).eval()

    @abstractmethod
    def extract_seg(self, image_obj, separator, box, ex_box, conf_threshold=0.2, min_px=500, save=False, basename=""):
        pass

    @abstractmethod
    def save_segments(self, img_pil, save_dir, group_name, basename):
        pass

    @abstractmethod
    def create_dim_external_mask(self, image_obj, targer_mask, save, group_name, basename, context_masks=None, opacity=0.25):
        pass

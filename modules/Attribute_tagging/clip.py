import open_clip
import torch
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_PROMPT, PROMPT_DESCRIPTION, TAXONOMY_HIERARCHY
from data.attributes_list import UNIVERSAL_ATTRIBUTES,UNIVERSAL_BOTTOM_ATTRIBUTES, UNIVERSAL_DRESS_ATTRIBUTES, UNIVERSAL_TOP_ATTRIBUTES
from utils.log_config import get_logger

logger = get_logger(__name__)

class FashionDataset(Dataset):
    """
    This class is responsible for preprocessign os the files before sending them to the tagging by FashionCLIP and FashionSigLip
    """
    def __init__(self, image_paths):
        self.image_paths = [str(p) for p in image_paths]
        self.background_color = (255,255,255)
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        This method is used to make sure the masks are in square format before sending to the CLIP as CLIP expects a square image (224X224). The method only pads to make the image square for the 224X224, it is left for CLIP to internally handle it.
        """
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            width, height = image.size
            square_size = max(width, height)

            new_img = Image.new("RGB", (square_size, square_size), self.background_color)
            x_offset = (square_size - width) // 2
            y_offset = (square_size - height) // 2

            if image.mode == 'RGBA':
                new_img.paste(image, (x_offset, y_offset), mask=image)
            else:
                new_img.paste(image, (x_offset, y_offset))

            return new_img, Path(img_path).name
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, Path(img_path).name

def collate_fn(batch):
    # Merge the images into batches based on category
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return [], []
    images, names = zip(*batch)
    return list(images), list(names)

class CLIP_SigLip_Attribute_Extractor:
    """
    This class is responsible for end to end steps required to run the tagging

    It starts from reading the different configurations for prompts, descriptions, taxonomies. Prepares the template and then maps the description for the prompts
    """
    def __init__(self, device='cuda', model_name="SigLip", descriptive=True):
        with open(CLIP_PROMPT, "r") as prompt_temp:
            config_prompt = json.load(prompt_temp)
        with open(PROMPT_DESCRIPTION, "r") as desc:
            config_desc = json.load(desc)
        with open(TAXONOMY_HIERARCHY, "r") as hierarchy:
            config_hierarchy = json.load(hierarchy)

        self.PROMPT_TEMPLATES = config_prompt["PROMPT_TEMPLATES"]
        if descriptive:
            self.clip_vocal_mapping = config_desc["PROMPT_DESCRIPTION"]
        else:
            self.clip_vocal_mapping = {}
        self.TAXONOMY_RULES = config_hierarchy["TAXONOMY_RULES"]
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if model_name == "SigLip":
            self.model, _, self.preprocess_val = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLip")
            self.tokeniser = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLip")
        elif model_name == "CLIP":
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.preprocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        else:
            logger.error(f"{model_name} not defined")
        self.model.to(self.device)
        self.model.eval()
        self.text_cache = {}
        self.model_name = model_name

    def group_imagesper_category(self, folder_path):
        """
        This method is used to group the images into categories based on the filename from segmentation
        """
        category_groups = defaultdict(list)
        labelmap = {"top": "upper_body", "outer_top": "upper_body"}
        all_files = [img for img in Path(folder_path).iterdir() if img.suffix.lower() in [".png", ".jpg"]]
        valid_mask_categories = list(self.TAXONOMY_RULES.keys())

        for img_path in all_files:
            stem_name = img_path.stem
            if "_p0" in stem_name:
                category = stem_name.split("_p0_")[-1]
                category = labelmap.get(category, category)
                if category in valid_mask_categories:
                    category_groups[category].append(img_path)
        return category_groups
    
    def get_text_features(self, category, attribute, candidates):
        """
        This method is used to precompute the text features into CLIP embedding space. It has loops to decide if descriptive prompts are to be put on embedding space or a direct prompts. Along with this it also uses the FashionSigLip or FashionCLIP to calculate the vectors in CLIP embedding space based on the type of model
        """
        cache_key = f"{category}_{attribute}"
        if cache_key in self.text_cache:
            return self.text_cache[cache_key]
        
        clean_mapping_candidates = []
        for map in candidates:
            if map in self.clip_vocal_mapping:
                clean_mapping_candidates.append(self.clip_vocal_mapping[map])
            else:
                clean_mapping_candidates.append(map.replace("_", " "))

        template = self.PROMPT_TEMPLATES.get(attribute, self.PROMPT_TEMPLATES["default"])
        prompts = [template.format(category=category, candidate=map) for map in clean_mapping_candidates]
        logger.info(prompts)
        if self.model_name == "SigLip":
            logger.info("Using SigLip model")
            text_inputs = self.tokeniser(prompts).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(text_inputs)
                features = features / features.norm(dim=-1, keepdim=True)
        elif self.model_name == "CLIP":
            logger.info("Using CLIP model")
            inputs = self.preprocess(text=prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

        self.text_cache[cache_key] = features
        return features
    
    def extract_attributes(self, folder_path, batch_size=32):
        """
        This method is used to run as an end to end methodology.
        It starts to group the images based on category
        Prepares the batch size using the torch dataloader
        BAsed on the model type to be used extracts the embeddings for images, compares it using cosine similarity and tags the attributes.
        """
        category_groups = self.group_imagesper_category(folder_path)
        all_results = []

        for mask_category, paths in category_groups.items():
            if not paths or mask_category not in self.TAXONOMY_RULES:
                continue
            logger.info(f"Processing {len(paths)} images for category: {mask_category}")
            config = self.TAXONOMY_RULES[mask_category]
            logger.info(config)
            dataset = FashionDataset(paths)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

            for images, names in dataloader:
                if not images:
                    continue

                if self.model_name == "SigLip":
                    logger.info("Using SigLip model")
                
                    image_tensor = torch.stack([self.preprocess_val(image) for image in images]).to(self.device)

                    with torch.no_grad():
                        image_features = self.model.encode_image(image_tensor)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                elif self.model_name == "CLIP":
                    logger.info("Using CLIP model")
                    img_inputs = self.preprocess(images=images, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        image_features = self.model.get_image_features(**img_inputs)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                batch_records = {name: {"filename": name, "main_category": mask_category} for name in names}

                if mask_category in ("skirt", "jeans", "trousers", "shorts", "pants"):
                    attr = UNIVERSAL_BOTTOM_ATTRIBUTES
                elif mask_category in ("upper_body"):
                    attr = UNIVERSAL_TOP_ATTRIBUTES
                elif mask_category in ("dress"):
                    attr = UNIVERSAL_DRESS_ATTRIBUTES
                else:
                    attr = UNIVERSAL_ATTRIBUTES
                
                for attribute, candidates in attr.items():
                    text_features = self.get_text_features(mask_category, attribute, candidates)
                    preds = (image_features @ text_features.T).softmax(dim=-1).argmax(dim=-1).cpu().tolist()

                    for index, name in enumerate(names):
                        batch_records[name][f"predicted_{attribute}"] = candidates[preds[index]]

                if not config["is_hierarchical"]:
                    # print("in here")
                    for attribute, candidates in config["attributes"].items():
                        # print("in here 2")
                        text_features = self.get_text_features(mask_category, attribute, candidates)
                        preds = (image_features @ text_features.T).softmax(dim=-1).argmax(dim=-1).cpu().tolist()
                        for index, name in enumerate(names):
                            batch_records[name][f"predicted_{attribute}"] = candidates[preds[index]]
                        # print(batch_records)
                        # break
                else:
                    routing_attribute = config["routing_attribute"]
                    routing_candidates = config["routing_candidates"]

                    main_features = self.get_text_features(mask_category, routing_attribute, routing_candidates)
                    route_preds = (image_features @ main_features.T).softmax(dim=-1).argmax(dim=-1).cpu().tolist()

                    for index, name in enumerate(names):
                        base_style = routing_candidates[route_preds[index]]
                        batch_records[name][f"predicted_{routing_attribute}"] = base_style

                        sub_config = config["sub_styles"].get(base_style, {})
                        single_image_feature = image_features[index].unsqueeze(0)

                        for subattribute, sub_candidates in sub_config.items():
                            sub_text_features = self.get_text_features(base_style, subattribute, sub_candidates)
                            sub_preds = (single_image_feature @ sub_text_features.T).softmax(dim=-1).argmax(dim=-1).item()
                            batch_records[name][f"predicted_{subattribute}"] = sub_candidates[sub_preds]

                    for name in names:
                        pred_print = batch_records[name].get("predicted_print_type")

                        if pred_print in ["plain"]:
                            batch_records[name]["predicted_pattern_scale"] = "none"
                all_results.extend(batch_records.values())
                logger.info(f"Completed for {mask_category}")
        return pd.DataFrame(all_results)

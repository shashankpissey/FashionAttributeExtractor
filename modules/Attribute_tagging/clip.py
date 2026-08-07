import open_clip
import torch
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

from modules.Attribute_tagging.color_tagging import ColourTagger
from config import CLIP_PROMPT, PROMPT_DESCRIPTION, TAXONOMY_HIERARCHY, GT_COLOUR_NAMES
from data.attributes_list import UNIVERSAL_ATTRIBUTES,UNIVERSAL_BOTTOM_ATTRIBUTES, UNIVERSAL_DRESS_ATTRIBUTES, UNIVERSAL_TOP_ATTRIBUTES
from utils.log_config import get_logger

logger = get_logger(__name__)

class CLIP_SigLip_Attribute_Extractor:
    """
    This class is responsible for end to end steps required to run the tagging

    It starts from reading the different configurations for prompts, descriptions, taxonomies. Prepares the template and then maps the description for the prompts
    """
    def __init__(self, device='cuda', model_name="SigLip", descriptive=True, colour_k=4):
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
        # Code similar to hugging face
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
        self.colour_k = colour_k
        self.colour_tagger = ColourTagger(GT_COLOUR_NAMES)

    def prepare_image_batch(self, image_path, background_colour=(255, 255, 255)):
        """
        This method is used to prepare the image. The image is read in RGBA and then only opaque picsls are obtained. The CLIP resize is left for its internal resize but the total image is made square by padding white pixels to shortest side.
        """
        try:
            image = Image.open(image_path).convert("RGBA")
            width, height = image.size
            square_size = max(width, height)

            new_img = Image.new("RGB", (square_size, square_size), background_colour)
            x_offset = (square_size - width) // 2
            y_offset = (square_size - height) // 2
            new_img.paste(image, (x_offset, y_offset), mask=image)
            return new_img, image, Path(image_path).name
        except Exception as e:
            logger.error(f"Padding failed with error {e}")
            return None, None, Path(image_path).name

    def batch_iteration(self, paths, batch_size):
        """
        This is a generator function that provides images per batch size based
        """
        for i in range(0, len(paths), batch_size):
            yield paths[i:i + batch_size]        
    
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
    
    def extract_attributes(self, folder_path, batch_size=32, file_name_to_save=""):
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

            for path_batch in self.batch_iteration(paths, batch_size):
                images, original_images, names = [], [], []

                for p in path_batch:
                    padded_image, original_image, name = self.prepare_image_batch(p)
                    if padded_image is not None:
                        images.append(padded_image)
                        original_images.append(original_image)
                        names.append(name)
                
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

                if file_name_to_save in ["segmented_a","segmented_b","segmented_c"]:
                    for index, name in enumerate(names):
                        colour_result = self.colour_tagger.extract_colours(original_images[index], self.colour_k)
                        batch_records[name]["predicted_colour"] = colour_result["predicted_colour"]
                        batch_records[name]["colour_percentage"] = colour_result["percentage"]
                        batch_records[name]["colour_delta_e"] = colour_result["delta_e_distance"]

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

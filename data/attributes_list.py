UNIVERSAL_TOP_ATTRIBUTES = {
    "formality_band": ["casual", "smart_casual","occasion", "polished"],
    "material": ["cotton", "denim", "satin", "silk", "leather","velvet", "chiffon", "linen", "synthetic", "rayon", "nylon", "polyester", "suede", "georgette", "knit"]
}

UNIVERSAL_BOTTOM_ATTRIBUTES = {
    "rise": ["low_rise", "mid_rise", "high_rise"],
    "drape": ["structured", "soft", "fluid", "crisp"],
    "texture": ["smooth", "textured", "highly_textured"],
    "pattern_scale": ["small", "large", "medium", "none"],
    "opacity_level": ["opaque", "sheer", "semi_sheer"],
    "visual_weight": ["light","medium","heavy"],
    "formality_band": ["casual", "smart_casual","occasion", "polished"],
    "seasonality": ["all_season","cold_season", "warm_season"],
    "material": ["cotton", "denim", "satin", "silk", "leather","velvet", "chiffon", "linen", "rayon", "nylon", "polyester", "suede", "georgette", "knit", "wool", "jersey", "lace", "mesh"]
}

UNIVERSAL_DRESS_ATTRIBUTES = {
    "drape": ["structured", "soft", "fluid", "crisp"],
    "texture": ["smooth", "textured", "highly_textured"],
                
}

UNIVERSAL_ATTRIBUTES = {
    "drape": ["structured", "soft", "none"],
    "pattern_scale": ["small", "large", "medium", "none"],
    "opacity_level": ["opaque", "sheer", "semi_sheer"],
    "visual_weight": ["light","medium","heavy"],
    "print_type": ["geometric", "abstract", "floral", "stripe", "animal", "graphic", "novelty", "dots", "plaid", "plain"]
            
}


# Evaluation

ATTRIBUTE_MAPPING = {
    "predicted_length": "tags.garment_length_class",
    "predicted_silhouette": "tags.silhouette_shape",
    "predicted_rise": "tags.rise_class",
    "predicted_drape": "tags.fabric_drape",
    "predicted_print_type": "tags.print_family",
    "predicted_opacity_level": "tags.opacity_level",
    "predicted_pattern_scale": "tags.pattern_scale",
    "predicted_formality_band": "tags.formality_band",
    "predicted_cut": "tags.cut",
    "predicted_texture": "tags.texture_prominence",
    "predicted_base_style": "category_x",
    "predicted_leg_shape": "tags.leg_shape",
    "predicted_closure_mode": "tags.closure_type",
    "predicted_waist_definition": "tags.waist_definition",
    "predicted_seasonality": "tags.seasonality",
    "predicted_sleeve_length": "tags.sleeve_length", 
    "predicted_sleeve_volume": "tags.sleeve_volume", 
    "predicted_neckline": "tags.neckline_shape",
    "predicted_visual_weight": "tags.visual_weight",
    "predicted_item_type": "item_type_x",
    "predicted_leg_shape": "tags.leg_shape",
    "predicted_material": "tags.material"
}
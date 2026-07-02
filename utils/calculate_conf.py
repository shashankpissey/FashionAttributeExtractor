import numpy as np

def calculate_conf_metadata(mask, box, h, w, edge_margin=3):
    xmin, ymin, xmax, ymax = box
    
    edges_cropped = []
    if ymin <= edge_margin:
        edges_cropped.append("top")
    if ymax >= h - edge_margin:
        edges_cropped.append("bottom")
    if xmin <= edge_margin:
        edges_cropped.append("left")
    if xmax >= w - edge_margin:
        edges_cropped.append("right")
 
    crop_status = "cropped_" + "_".join(edges_cropped) if edges_cropped else "full_garment"
 
    mask_pixels = int(np.count_nonzero(mask))
    bbox_area = max(1, (ymax - ymin) * (xmax - xmin))
    density = round(mask_pixels / bbox_area, 3)
    area_ratio = round(bbox_area / (h * w), 4)
    
    return {
        "crop_status": crop_status,
        "edges_cropped": edges_cropped,
        "density": density,
        "area_ratio": area_ratio,
        "is_cutoff": bool(edges_cropped)
    }
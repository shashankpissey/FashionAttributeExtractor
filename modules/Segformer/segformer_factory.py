from .base_segmenter import BaseSegformer
from .segmenter import Segmenter
from .segmenter_all import SegmenterAll
from config import SEGMENTATION_MODEL, SEGMENTER_MODEL


def segformer_factory(condition, device):
    if condition == "ALL":
        return SegmenterAll(model=SEGMENTER_MODEL, device=device)
    return Segmenter(model=SEGMENTATION_MODEL, device=device)

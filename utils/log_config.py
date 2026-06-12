import logging
from datetime import datetime as dt


file_name = f"feature_extractor_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logging.basicConfig(
    filename=file_name,
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name):
    return logging.getLogger(name)
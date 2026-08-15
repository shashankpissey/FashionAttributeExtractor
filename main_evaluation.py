from evaluation.Evaluate_IoU import Evaluate_IoU
import json


if __name__ == "__main__":
    evaluate = Evaluate_IoU()
    with open("D:/Dissertation/Final/test.json", "r") as f:
          split_filenames = set(json.load(f))
    evaluate.evaluate_pipeline(split_filenames)
import cv2
import os
import traceback
from config import CLASS_ID


from ultralytics import YOLO

class PersonDetector:
    def __init__(self, yolo_model, device):
        print("initialisation")
        self.det_model = YOLO(yolo_model) 
        self.device = device
        print("Object detection model loaded")

    def detect_person(self, image_array):
        
        try:
            result = self.det_model(image_array)[0]

            person_box = None
            max_area = 0

            for box in result.boxes:
                if int(box.cls) == CLASS_ID:
                    x_start, y_start, x_end, y_end = [int(pix) for pix in box.xyxy[0].tolist()]
                    area = (x_end - x_start) * (y_end - y_start)
                    if area > max_area:
                        max_area = area
                        person_box = [x_start, y_start, x_end, y_end]
            return person_box

        except FileNotFoundError:
            raise FileNotFoundError
        except Exception:
            raise Exception
    

    def crop_person(self, image_array, person_box, expand_perc=0.45):
        try:
            person_crop = cv2.imread(image_array)
            if person_crop is None:
                print("No person Found")
                return None
            
            person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            x_start, y_start, x_end, y_end = map(int, person_box)
            img_height, img_width = person_crop.shape[:2]
            box_width = x_end - x_start
            box_height = y_end - y_start
            pad_w = int(box_width * expand_perc)
            pad_h = int(box_height * expand_perc)
            new_x_start = max(0, x_start - pad_w)
            new_y_start = max(0, y_start - pad_h)
            new_x_end = min(img_width, x_end + pad_w)
            new_y_end = min(img_height, y_end + pad_h)
            
            return person_crop[new_y_start:new_y_end, new_x_start:new_x_end]
        except Exception:
            print(traceback.format_exc())
    

    def save_person(self, image_array, output_path, person_base):
        print(f"output path {output_path}")
        output_dir = os.path.dirname(output_path)
        print(output_dir)
        output_dir1 = output_dir+"/"+person_base+".png" 
        print(f"output dir1 {output_dir1}")
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return cv2.imwrite(output_dir1, image_bgr)
    

    def detect_all_people(self, image_array, conf_threshold=0.5):
        try:
            result = self.det_model(image_array)[0]
            person_boxes = []

            for box in result.boxes:
                box_confidence = float(box.conf[0])
                if box_confidence >= float(conf_threshold) and int(box.cls) == CLASS_ID:
                    print(box_confidence)
                    x_start, y_start, x_end, y_end = [int(pix) for pix in box.xyxy[0].tolist()]

                    if (x_end-x_start) * (y_end-y_start) > 500:
                        person_boxes.append([x_start, y_start, x_end, y_end])
            return person_boxes
        except Exception as e:
            raise e
        
    def crop_person_ex(self, image_array, person_box, expand_perc=0.45):
        try:
            person_crop = cv2.imread(image_array)
            if person_crop is None:
                print("No person Found")
                return None, None
            
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            x_start, y_start, x_end, y_end = map(int, person_box)
            img_height, img_width = person_crop_rgb.shape[:2]
            box_width = x_end - x_start
            box_height = y_end - y_start
            pad_w = int(box_width * expand_perc)
            pad_h = int(box_height * expand_perc)
            new_x_start = max(0, x_start - pad_w)
            new_y_start = max(0, y_start - pad_h)
            new_x_end = min(img_width, x_end + pad_w)
            new_y_end = min(img_height, y_end + pad_h)

            ex_crop = person_crop_rgb[new_y_start:new_y_end, new_x_start:new_x_end]
            ex_coord = [new_x_start, new_y_start, new_x_end, new_y_end] 
            
            return ex_crop, ex_coord
        except Exception:
            print(traceback.format_exc())
            return None, None
    
        




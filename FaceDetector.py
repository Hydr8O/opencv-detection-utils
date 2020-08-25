import cv2
import numpy as np
from .ImageForDetection import ImageForDetection

class FaceDetector:
    def __init__(self, proto_path, model_path, confidence):
        self._detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self._boxes = []
        self._confidence = confidence
        
    def detect_image(self, image, crop=(800, 800, 700, 200)):
        crop_x, crop_y, crop_width, crip_height = crop
        image = ImageForDetection(image)
        cropped = image.crop(crop_x, crop_y, crop_width, crip_height)
        (original_height, original_width) = cropped.get_shape()
        blob = cropped.get_blob(mean=(104.0, 177.0, 123.0), size=(300, 300), swapRB=False, scalefactor=1.0)
        self._detector.setInput(blob)
        detections = self._detector.forward()
        self._boxes = []
        for i in range(0, detections.shape[2]):
            box = (detections[0, 0, i, 3:7] * np.array([original_width, original_height, original_width, original_height])).astype('int')
            x, y, width, height = box
            difference_x = width - x
            difference_y = height - y
            x = x + crop_x
            y = y + crop_y   
            width = x + difference_x
            height = y + difference_y
            box = [x, y, width, height]
            confidence = detections[0, 0, i, 2]

            if (confidence > self._confidence):
                self._boxes.append(box)
                image.draw_border_box(box, 'face', (255, 0, 0), end_box_coordinates=True)
        return image.get_image()
                
    def is_object(self):
        if len(self._boxes) == 0:
            return False
        else:
            return True

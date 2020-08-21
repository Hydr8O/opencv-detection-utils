import cv2
import numpy as np
class OutputInterpreter():
    def __init__(self, outputs, image_size, classes, colors, confidence_threshold=.4):
        self._confidence_threshold = confidence_threshold
        self._outputs = outputs
        self._classes = classes
        self._colors = colors
        self._boxes = []
        self._confidences = []
        self._class_ids = []
        self._width, self._height = image_size
        self._interpret_outputs()
        try:
            self._nms_indexes = cv2.dnn.NMSBoxes(
                bboxes=self._boxes, 
                scores=self._confidences, 
                score_threshold=self._confidence_threshold, 
                nms_threshold=0.2
            ).flatten()   
        except AttributeError:
            self._nms_indexes = []
        
    def _interpret_outputs(self):
        for out in self._outputs:
            for detection in out:
                self._interpret_detection(detection)
                
                    
    def _interpret_detection(self, detection):
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > self._confidence_threshold:
            center_x = int(detection[0] * self._width)
            center_y = int(detection[1] * self._height)
            box_width = int(detection[2] * self._width)
            box_height = int(detection[3] * self._height)      
            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)
            self._boxes.append([x, y, box_width, box_height])
            self._class_ids.append(class_id)
            self._confidences.append(float(confidence))
     
    def get_detections(self):
        filtered_boxes = []
        labels = []
        colors = []
        for index in self._nms_indexes:
            labels.append(self._classes[self._class_ids[index]])
            filtered_boxes.append(self._boxes[index])   
            colors.append(self._colors[self._class_ids[index]])    
        return (filtered_boxes, labels, colors)
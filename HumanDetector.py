import cv2
from ImageForDetection import ImageForDetection
   
class HumanDetector:
    def __init__(
        self, 
        win_stride=(8, 8), 
        padding=(16, 16), 
        scale=1.1, 
        hit_threshold=.6
    ):
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold
        
        
    def detect_image(self, image):
        hog = cv2.HOGDescriptor() 
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
        image = ImageForDetection(image)
        (boxes, _) = hog.detectMultiScale(
            image.get_image(), 
            winStride=self.win_stride, 
            padding=self.padding, 
            scale=self.scale, 
            hitThreshold=self.hit_threshold
        ) 
        
        for box in boxes:
            image.draw_border_box(box, 'person', (0, 255, 0))
            
        return image
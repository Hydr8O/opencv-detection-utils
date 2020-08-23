import cv2
from .ImageForDetection import ImageForDetection
   
class HumanDetector:
    def __init__(
        self, 
        win_stride=(8, 8), 
        padding=(16, 16), 
        scale=1.1, 
    ):
        self._win_stride = win_stride
        self._padding = padding
        self._scale = scale
        self._boxes = []
        
        
    def detect_image(self, image):
        hog = cv2.HOGDescriptor() 
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
        image = ImageForDetection(image)
        (self._boxes, _) = hog.detectMultiScale(
            image.get_image(), 
            winStride=self._win_stride, 
            padding=self._padding, 
            scale=self._scale, 
        ) 
        
        for box in self._boxes:
            image.draw_border_box(box, 'person', (0, 255, 0))
            
        return image.get_image()
    
    def is_human(self):
        if len(self._boxes) == 0:
            return False
        else:
            return True

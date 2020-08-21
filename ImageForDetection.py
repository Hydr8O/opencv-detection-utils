import cv2
class ImageForDetection:
    def __init__(self, image):
        self._image = image
        self._height, self._width, _ = self._image.shape
        
    def show_image(self):
        cv2.imshow('Image', self._image)
    
    def get_blob(self, size=(410, 410)):
        blob = cv2.dnn.blobFromImage(
            self._image, 
            scalefactor=1/255.0, 
            size=size, 
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        return blob
    
    def get_image(self):
        return self._image
    
    def get_width(self):
        return self._width
    
    def get_height(self):
        return self._height
    
    def draw_border_box(self, box, label, color):
        FONT = cv2.FONT_HERSHEY_PLAIN
        x, y, box_width, box_height = box
        top_left_coordinates = (x, y)
        rectangle_color = color
        cv2.rectangle(
            self._image, 
            top_left_coordinates,
            (x + box_width, y + box_height),
            rectangle_color,
            2
        )
        
        cv2.putText(
            self._image,
            label,
            (x, y - 10),
            FONT,
            2,
            color,
            2
        )
    
    def save(self, file_name):
        cv2.imwrite(file_name, self._image)
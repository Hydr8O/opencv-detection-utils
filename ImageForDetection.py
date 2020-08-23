import cv2
class ImageForDetection:
    def __init__(self, image):
        self._image = image
        self._height, self._width, _ = self._image.shape
        
    def show_image(self):
        cv2.imshow('Image', self._image)
    
    def resize(self, size):
        self._image = cv2.resize(self._image, size)
    
    def get_blob(self, size=(410, 410), swapRB=True, mean=(0, 0, 0), scalefactor=1/255.0, crop=False):
        blob = cv2.dnn.blobFromImage(
            self._image, 
            scalefactor=scalefactor, 
            size=size, 
            mean=mean,
            swapRB=swapRB,
            crop=crop
        )
        return blob
    
    def get_image(self):
        return self._image
    
    def crop(self, x, y, w, h):
        return ImageForDetection(self._image[y:y+h, x:x+w])
    
    def get_width(self):
        return self._width
    
    def get_height(self):
        return self._height
    
    def get_shape(self):
        return self._height, self._width
    
    def draw_border_box(self, box, label, color, end_box_coordinates=False):
        FONT = cv2.FONT_HERSHEY_PLAIN
        x, y, box_width, box_height = box
        top_left_coordinates = (x, y)
        if end_box_coordinates == False:
            end_x, end_y = (x + box_width, y + box_height)
        else:
            end_x, end_y = (box_width, box_height)
        rectangle_color = color
        cv2.rectangle(
            self._image, 
            top_left_coordinates,
            (end_x, end_y),
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

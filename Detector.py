from .ImageForDetection import ImageForDetection
from .OutputInterpreter import OutputInterpreter
import numpy as np

class Detector:
	def __init__(self, model, classes_path):
		self.model = model
		layer_names = model.getLayerNames()
		self.out_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
		self.classes = self._read_classes(classes_path)
		self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
	def _read_classes(self, classes_path):
		classes = []
		with open(classes_path, 'r') as file:
			for line in file:
				classes.append(line.strip())
		return classes
	def detect_image(self, image):
		image = ImageForDetection(image)
		blob = image.get_blob()
		self.model.setInput(blob)
		outputs = self.model.forward(self.out_layers)
		interpreter = OutputInterpreter(
			outputs,
			(image.get_width(), image.get_height()),
			self.classes,
			self.colors
		)

		boxes, labels, colors = interpreter.get_detections()

		for index, box in enumerate(boxes):
			color = colors[index]
			label = labels[index]
			image.draw_border_box(box, label, color)

		return image
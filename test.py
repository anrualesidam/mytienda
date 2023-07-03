from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

#model = YOLO('yolov8n.pt')
model = YOLO('yolov8s.pt')
#model = YOLO('yolov8l.pt')
#model = YOLO('yolov8m.pt')
#model = YOLO('yolov8x.pt')
results = model.predict(source='0', classes=0 , show=True)
print(results)

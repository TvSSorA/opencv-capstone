from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.track(source='0', persist=True, show=True, conf=0.6, classes=[0])
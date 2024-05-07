from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.predict(source='rtsp://192.168.5.157/live/0/MAIN', show=True, conf=0.6, classes=[0])

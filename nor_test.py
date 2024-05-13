import cv2
from ultralytics import YOLO
from norfair import Detection, Tracker
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Initialize Norfair tracker
distance_function = "euclidean"
distance_threshold = 25
tracker = Tracker(distance_function=distance_function, distance_threshold=distance_threshold)

# Kalman filter parameters
tracker.measurement_noise = 5.0
tracker.process_noise = 0.1

# Open the RTSP stream
cap = cv2.VideoCapture('rtsp://192.168.5.157/live/0/MAIN')

# Frame counter
counter = 0

# Number of frames to skip
skip_frames = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    counter += 1
    if counter % skip_frames != 0:
        continue

    results = model(frame)

    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            if int(cls) == 0 and conf >= 0.5:
                centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                detections.append(Detection(points=centroid, data={"scores": conf, "bbox": [x1, y1, x2, y2]}))

    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        # Access data from the last detection associated with the tracked object
        bbox = obj.last_detection.data["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj.id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Real-Time Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
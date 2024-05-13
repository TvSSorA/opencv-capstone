from ultralytics import YOLO
import cv2
import math
from sort import *
cap = cv2.VideoCapture(0)  # For RTSP video source
model = YOLO('yolov8l.pt')

classNames = ["person"]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

frame_count = 0
skip_frames = 5  # Change this to skip more or less frames

# Set to store unique IDs
unique_ids = set()

while True:
    success, img = cap.read()

    # Skip frames
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Reduce resolution
    img = cv2.resize(img, (1280, 720))  # Change this to your desired resolution

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # Only process the detection if the class is "person"
            if cls == 0 and conf > 0.6:  # 0 is the class index for "person" in COCO
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Add the ID to the set of unique IDs
        unique_ids.add(id)

        w, h = x2 - x1, y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, f'ID: {int(id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the number of unique people on the video screen
    cv2.putText(img, f'Number of unique people: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
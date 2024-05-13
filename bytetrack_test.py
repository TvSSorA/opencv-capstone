import cv2
import supervision as sv
from supervision import ColorLookup
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
tracker = sv.ByteTrack()  # Initialize ByteTrack
box_annotator = sv.BoundingBoxAnnotator(color_lookup=ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Frame counter
counter = 0

# Number of frames to skip
skip_frames = 7

def process_frame(frame):
    results = model(frame)[0]  # Perform object detection
    detections = sv.Detections.from_ultralytics(results)

    # Filter only humans
    human_detections = detections[detections.class_id == 0]

    # Add a confidence score threshold
    confidence_threshold = 0.5  # Set your desired confidence threshold here
    human_detections = human_detections[human_detections.confidence > confidence_threshold]

    human_detections = tracker.update_with_detections(human_detections)  # Update tracker with human detections

    labels = [f"#{tracker_id}" for tracker_id in human_detections.tracker_id]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=human_detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=human_detections, labels=labels)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=human_detections)
    return annotated_frame

# Open RTSP stream
cap = cv2.VideoCapture("rtsp://192.168.5.157/live/0/MAIN")

if not cap.isOpened():
    print("Error opening video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    counter += 1
    if counter % skip_frames != 0:
        continue

    vi = process_frame(frame)
    annotated_frame = cv2.resize(vi, (960, 540))
    cv2.imshow("Tracking Results (Humans Only)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
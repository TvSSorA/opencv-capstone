import cv2
import supervision as sv
from supervision import ColorLookup
from ultralytics import YOLO
import uuid
import os
from datetime import datetime
import time


start_time = time.time()
# Initialize the YOLO model
model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()  # Initialize ByteTrack

# Set up annotators
box_annotator = sv.BoundingBoxAnnotator(color_lookup=ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Frame counter
counter = 0

# Number of frames to skip
skip_frames = 1

# Source video path
SOURCE_VIDEO_PATH = 'cctv.mp4'

# Dictionary to map tracker_id to UUID
tracker_id_to_uuid = {}

# Define the directory to save the cropped images
output_dir = 'cropped_images'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set to store unique IDs for which images have been saved
saved_images_ids = set()


def get_uuid_for_tracker_id(tracker_id):
    if tracker_id not in tracker_id_to_uuid:
        # Generate a UUID and take the first 8 characters
        tracker_id_to_uuid[tracker_id] = str(uuid.uuid4()).replace("-", "")
    return tracker_id_to_uuid[tracker_id]


def process_frame(results, frame):
    detections = sv.Detections.from_ultralytics(results)

    # Filter only humans
    human_detections = detections[detections.class_id == 0]
    if not human_detections:
        return frame
    human_detections = tracker.update_with_detections(human_detections)  # Update tracker with human detections

    # Generate or retrieve UUIDs for each tracker ID
    labels = [f"#{get_uuid_for_tracker_id(tracker_id)}" for tracker_id in human_detections.tracker_id]

    # Get the minimum length of the two lists
    min_length = min(len(human_detections.xyxy.tolist()), len(labels))

    # Iterate through the bounding boxes
    for i in range(min_length):  # Only iterate over valid indices
        box = human_detections.xyxy.tolist()[i]
        # Check if the image for this ID has already been saved
        if labels[i] in saved_images_ids:
            continue

        x1, y1, x2, y2 = box
        # Crop the object using the bounding box coordinates
        crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
        # Get the current date and time
        now = datetime.now()
        # Format the date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        # Save the cropped object as an image in the defined directory
        cv2.imwrite(os.path.join(output_dir, labels[i] + '@' + date_time + '.jpg'), crop_object)

        # Add the ID to the set of saved images IDs
        saved_images_ids.add(labels[i])

    # Annotate the frame with bounding boxes, labels, and traces
    annotated_frame = box_annotator.annotate(frame.copy(), detections=human_detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=human_detections, labels=labels)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=human_detections)
    return annotated_frame


# Open RTSP stream and start prediction
results = model.predict(source=SOURCE_VIDEO_PATH, show=False, stream=True, classes=[0])

for result in results:
    frame = result.orig_img  # Get the original frame
    counter += 1
    if counter % skip_frames != 0:
        continue

    annotated_frame = process_frame(result, frame)
    if annotated_frame is None:
        continue
    annotated_frame = cv2.resize(annotated_frame, (960, 540))
    cv2.imshow("Tracking Results", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time - start_time
print(f"The total time taken by the code is {total_time} seconds.")
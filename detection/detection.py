# detection/detection.py

import cv2
import supervision as sv
from supervision import ColorLookup
from ultralytics import YOLO
import uuid
import os
from datetime import datetime
import threading
import queue
from loguru import logger
from pymongo import MongoClient

# Import utils
from .utils import logger

# MongoDB setup
client = MongoClient('mongodb+srv://adam123:tntguy123@vnmc-database.r8b4uv0.mongodb.net/')
db = client['capstone-project']
collection = db['images']
devices_collection = db['devices']

# Initialize the YOLO model
model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()  # Initialize ByteTrack

# Set up annotators
box_annotator = sv.BoundingBoxAnnotator(color_lookup=ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Directories to save images
images_dir = 'outputs'
base_output_dir = 'cropped_images'
base_annotated_output_dir = 'annotated_images'
base_whole_frame_dir = 'whole_frames'

# Create base directories if they don't exist
os.makedirs(images_dir, exist_ok=True)


# Store unique IDs and their detection information
tracker_id_to_uuid = {}
saved_images_ids = set()
current_uuids = set()

# Dictionary for frame queues and threads for each camera
frame_queues = {}
capture_threads = {}
process_threads = {}
stop_events = {}  # Dictionary to store stop events for each camera
active_cameras = {}  # Dictionary to keep track of active cameras


def get_uuid_for_tracker_id(tracker_id):
    if tracker_id not in tracker_id_to_uuid:
        # Generate a UUID
        new_uuid = str(uuid.uuid4()).replace("-", "")
        tracker_id_to_uuid[tracker_id] = new_uuid
        # Insert initial document to MongoDB
        collection.insert_one({
            'uuid': new_uuid,
            'entry_timestamp': datetime.now(),
            'exit_timestamp': None,
            'date': datetime.now().date().isoformat()
        })
    return tracker_id_to_uuid[tracker_id]


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_frame(device_id, frame, results, update_callback=None):
    detections = sv.Detections.from_ultralytics(results)

    # Filter only humans
    human_detections = detections[detections.class_id == 0]
    human_detections = tracker.update_with_detections(human_detections)  # Update tracker with human detections

    # Generate or retrieve UUIDs for each tracker ID
    labels = [f"#{get_uuid_for_tracker_id(tracker_id)}" for tracker_id in human_detections.tracker_id]

    # Get the minimum length of the two lists
    min_length = min(len(human_detections.xyxy.tolist()), len(labels))

    new_detected_uuids = set()

    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(images_dir, base_output_dir, device_id, current_date)
    annotated_output_dir = os.path.join(images_dir, base_annotated_output_dir, device_id, current_date)
    whole_frame_dir = os.path.join(images_dir, base_whole_frame_dir, device_id, current_date)

    ensure_directory_exists(images_dir)
    ensure_directory_exists(output_dir)
    ensure_directory_exists(annotated_output_dir)
    ensure_directory_exists(whole_frame_dir)

    # Iterate through the bounding boxes
    for i in range(min_length):  # Only iterate over valid indices
        box = human_detections.xyxy.tolist()[i]
        uuid_label = labels[i]

        new_detected_uuids.add(uuid_label)

        # Check if the image for this ID has already been saved
        if uuid_label not in saved_images_ids:
            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
            # Get the current date and time
            now = datetime.now()
            # Format the date and time
            date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            # Save the cropped object as an image in the defined directory
            cv2.imwrite(os.path.join(output_dir, uuid_label + '-' + date_time + '.jpg'), crop_object)
            # Save the whole frame
            cv2.imwrite(os.path.join(whole_frame_dir, uuid_label + '-whole-' + date_time + '.jpg'), frame)
            # Add the ID to the set of saved images IDs
            saved_images_ids.add(uuid_label)

    # Update exit timestamp for UUIDs no longer detected
    for uuid_label in current_uuids - new_detected_uuids:
        collection.update_one(
            {'uuid': uuid_label},
            {'$set': {'exit_timestamp': datetime.now()}}
        )

    # Check if there are new UUIDs detected
    if new_detected_uuids - current_uuids:
        current_uuids.update(new_detected_uuids)
        # Annotate the frame with bounding boxes, labels, and traces
        annotated_frame = box_annotator.annotate(frame.copy(), detections=human_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=human_detections, labels=labels)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections=human_detections)

        # Save the annotated frame
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        cv2.imwrite(os.path.join(annotated_output_dir, 'annotated-' + date_time + '.jpg'), annotated_frame)

        # Call the update callback
        if update_callback:
            update_callback({
                "uuid": uuid_label,
                "entry_timestamp": datetime.now().isoformat(),
                "image_path": os.path.join(whole_frame_dir, uuid_label + '-whole-' + date_time + '.jpg'),
                "annotated_image_path": os.path.join(annotated_output_dir, 'annotated-' + date_time + '.jpg')
            })

        return annotated_frame

    return None


def capture_frames(rtsp_url, device_id):
    logger.info(f"Starting frame capture for device {device_id} with URL {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened() and not stop_events[device_id].is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queues[device_id].full():
            continue
        frame_queues[device_id].put(frame)
    cap.release()
    logger.info(f"Stopped frame capture for device {device_id}")


def detect_and_process_frames(device_id, update_callback=None):
    logger.info(f"Starting frame processing for device {device_id}")
    while device_id in frame_queues and not stop_events[device_id].is_set():
        if frame_queues[device_id].empty():
            continue
        frame = frame_queues[device_id].get()
        results = model.predict(source=frame, show=False, stream=False, classes=[0], imgsz=640)
        for result in results:
            annotated_frame = process_frame(device_id, frame, result, update_callback)
            if annotated_frame is not None:
                logger.info(f"New frame processed and saved for device {device_id}.")
    logger.info(f"Stopped frame processing for device {device_id}")


def get_rtsp_url(device_id):
    device = devices_collection.find_one({"_id": device_id})
    if device and "rtsp_url" in device:
        return device["rtsp_url"]
    return None


def start_detection(device_id, update_callback=None):
    rtsp_url = get_rtsp_url(device_id)
    if not rtsp_url:
        logger.error(f"RTSP URL not found for device with ID: {device_id}")
        return

    frame_queues[device_id] = queue.Queue(maxsize=10)
    stop_events[device_id] = threading.Event()  # Initialize stop event for this device

    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, device_id))
    process_thread = threading.Thread(target=detect_and_process_frames, args=(device_id, update_callback))

    capture_threads[device_id] = capture_thread
    process_threads[device_id] = process_thread

    # Update device status to "connected"
    devices_collection.update_one({"_id": device_id}, {"$set": {"status": "connected"}})

    capture_thread.start()
    process_thread.start()

    active_cameras[device_id] = rtsp_url  # Track active camera
    logger.info(f"Started detection for device {device_id}. Total active devices: {len(capture_threads)}")

def stop_detection(device_id):
    logger.info(f"Attempting to stop detection for device {device_id}")

    if device_id in stop_events:
        stop_events[device_id].set()  # Signal threads to stop

    if device_id in capture_threads:
        logger.debug(f"Stopping capture thread for device {device_id}")
        capture_threads[device_id].join(timeout=5)  # Allow some time for the thread to stop
        capture_threads.pop(device_id)
    else:
        logger.warning(f"No capture thread found for device {device_id}")

    if device_id in process_threads:
        logger.debug(f"Stopping process thread for device {device_id}")
        process_threads[device_id].join(timeout=5)  # Allow some time for the thread to stop
        process_threads.pop(device_id)
    else:
        logger.warning(f"No process thread found for device {device_id}")

    if device_id in frame_queues:
        logger.debug(f"Clearing frame queue for device {device_id}")
        frame_queues.pop(device_id)
    else:
        logger.warning(f"No frame queue found for device {device_id}")

    if device_id in active_cameras:
        logger.debug(f"Removing active camera entry for device {device_id}")
        active_cameras.pop(device_id)
    else:
        logger.warning(f"No active camera entry found for device {device_id}")

    if device_id in stop_events:
        logger.debug(f"Removing stop event for device {device_id}")
        stop_events.pop(device_id)

    # Update device status to "disconnected"
    devices_collection.update_one({"_id": device_id}, {"$set": {"status": "disconnected"}})
    logger.info(f"Stopped detection for device {device_id}. Total active devices: {len(capture_threads)}")


def list_active_cameras():
    return list(active_cameras.keys())

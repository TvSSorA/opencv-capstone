import cv2
import supervision as sv
from ultralytics import YOLO
import uuid
import os
import time
from datetime import datetime
import threading
import queue
from loguru import logger
import asyncio

from config import Config
from detection.db_interactions import save_basic_image_metadata, update_device_status, get_rtsp_url
from detection.image_processing import ensure_directory_exists, save_images, create_annotated_frames, send_update_to_clients

# Initialize the YOLO model
model = YOLO(Config.MODEL_WEIGHTS)
tracker = sv.ByteTrack(
    track_activation_threshold=Config.TRACK_THRESH,
    lost_track_buffer=Config.TRACK_SECONDS * Config.FRAME_RATE,
    minimum_matching_threshold=Config.MATCH_THRESH,
    frame_rate=Config.FRAME_RATE
)

# Set up annotators
box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
heat_map_annotator = sv.HeatMapAnnotator(
    position=sv.Position.BOTTOM_CENTER,
    opacity=Config.HEATMAP_ALPHA,
    radius=Config.RADIUS,
    kernel_size=25,
    top_hue=0,
    low_hue=125
)

# Directories to save images
images_dir = Config.OUTPUT_DIR
os.makedirs(images_dir, exist_ok=True)

tracker_id_to_uuid = {}
saved_images_ids = set()
current_uuids = set()

frame_queues = {}
capture_threads = {}
process_threads = {}
stop_events = {}
active_cameras = {}

def get_uuid_for_tracker_id(tracker_id):
    if tracker_id not in tracker_id_to_uuid:
        new_uuid = str(uuid.uuid4())
        tracker_id_to_uuid[tracker_id] = new_uuid
    return tracker_id_to_uuid[tracker_id]

async def send_update(data, update_callback):
    await update_callback(data)

async def process_frame(device_id, frame, results, update_callback=None):
    try:
        logger.info(f"Processing frame for device {device_id}")
        detections = sv.Detections.from_ultralytics(results)
        human_detections = detections[detections.class_id == 0]
        human_detections = tracker.update_with_detections(human_detections)
        labels = [f"{get_uuid_for_tracker_id(tracker_id)}" for tracker_id in human_detections.tracker_id]
        min_length = min(len(human_detections.xyxy.tolist()), len(labels))
        new_detected_uuids = set()
        current_date = datetime.now().strftime("%Y-%m-%d")

        output_dir = os.path.join(images_dir, 'cropped_images', device_id, current_date)
        annotated_output_dir = os.path.join(images_dir, 'annotated_images', device_id, current_date)
        whole_frame_dir = os.path.join(images_dir, 'whole_frames', device_id, current_date)
        heatmap_output_dir = os.path.join(images_dir, 'heatmap_outputs', device_id, current_date)
        single_box_annotated_dir = os.path.join(images_dir, 'single_box_annotated', device_id, current_date)

        ensure_directory_exists(output_dir)
        ensure_directory_exists(annotated_output_dir)
        ensure_directory_exists(whole_frame_dir)
        ensure_directory_exists(heatmap_output_dir)
        ensure_directory_exists(single_box_annotated_dir)

        uuid_label = None  # Initialize uuid_label to None
        whole_frame_path = None  # Initialize whole_frame_path to None
        crop_path = None  # Initialize crop_path to None
        single_box_path = None  # Initialize single_box_path to None
        date_time = None  # Initialize date_time to None

        for i in range(min_length):
            box = human_detections.xyxy[i]
            uuid_label = labels[i]
            new_detected_uuids.add(uuid_label)
            logger.info(f"Detected person {uuid_label} at position {box}")

            if uuid_label not in saved_images_ids:
                logger.info(f"New person detected: {uuid_label}. Saving images.")
                crop_path, whole_frame_path, single_box_path, date_time = save_images(
                    frame, box, uuid_label, output_dir, whole_frame_dir, single_box_annotated_dir, human_detections
                )

                save_basic_image_metadata(uuid_label, device_id, crop_path, int(datetime.now().timestamp() * 1000))
                saved_images_ids.add(uuid_label)

        if new_detected_uuids - current_uuids:
            current_uuids.update(new_detected_uuids)

        annotated_frame = create_annotated_frames(frame, human_detections, labels, annotated_output_dir, heatmap_output_dir, box_annotator, label_annotator, trace_annotator, heat_map_annotator)

        # Only call send_update_to_clients if uuid_label and whole_frame_path have been assigned
        if uuid_label and whole_frame_path and crop_path:
            await send_update_to_clients(device_id, annotated_frame, uuid_label, crop_path, update_callback)

    except Exception as e:
        logger.error(f"Error processing frame for device {device_id}: {e}")

def capture_frames(rtsp_url, device_id):
    logger.info(f"Starting frame capture for device {device_id} with URL {rtsp_url}")
    retries = 0
    connected = False

    while retries < Config.MAX_RETRIES:
        start_time = time.time()
        cap = None

        # Create a thread to handle the VideoCapture opening
        def open_video_capture():
            nonlocal cap
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        capture_thread = threading.Thread(target=open_video_capture)
        capture_thread.start()
        capture_thread.join(timeout=10)  # Timeout after 10 seconds

        # Check if the connection was successful
        if cap is not None and cap.isOpened():
            connected = True
            logger.info(f"Successfully connected to stream for device {device_id}")
            break
        else:
            logger.error(f"Failed to open stream for device {device_id}, retrying... ({retries + 1}/{Config.MAX_RETRIES})")
            retries += 1
            time.sleep(Config.RETRY_DELAY)

    if not connected:
        logger.error(f"Max retries reached for device {device_id}. Stopping capture.")
        return False

    # Custom timeout handling for reading frames
    while cap.isOpened() and not stop_events[device_id].is_set():
        ret, frame = cap.read()
        elapsed_time = time.time() - start_time

        if not ret or elapsed_time > 10:  # 10 seconds timeout for reading frames
            logger.warning(f"Frame read timeout for device {device_id}, retrying... ({retries + 1}/{Config.MAX_RETRIES})")
            retries += 1
            time.sleep(Config.RETRY_DELAY)
            break

        if frame_queues[device_id].full():
            continue
        frame_queues[device_id].put(frame)

    cap.release()
    logger.info(f"Stopped frame capture for device {device_id}")
    update_device_status(device_id, "offline")

def detect_and_process_frames(device_id, update_callback=None):
    logger.info(f"Starting frame processing for device {device_id}")
    last_frame_time = time.time()
    while device_id in frame_queues and not stop_events[device_id].is_set():
        if frame_queues[device_id].empty():
            continue

        current_time = time.time()
        elapsed_time = current_time - last_frame_time
        if elapsed_time < 1 / Config.FRAME_RATE:
            time.sleep(1 / Config.FRAME_RATE - elapsed_time)

        frame = frame_queues[device_id].get()
        results = model.predict(source=frame, show=False, stream=True, classes=[0], imgsz=640)
        for result in results:
            asyncio.run(process_frame(device_id, frame, result, update_callback))
        last_frame_time = current_time

    logger.info(f"Stopped frame processing for device {device_id}")


def start_detection(device_id, update_callback=None):
    rtsp_url = get_rtsp_url(device_id)
    if not rtsp_url:
        logger.error(f"RTSP URL not found for device with ID: {device_id}")
        return

    frame_queues[device_id] = queue.Queue(maxsize=10)
    stop_events[device_id] = threading.Event()

    # Start the capture thread and wait for it to complete
    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, device_id))
    capture_thread.start()
    capture_thread.join()

    # Check if the connection was successful by verifying if the device is connected
    if device_id in frame_queues and not stop_events[device_id].is_set():
        if frame_queues[device_id].empty():  # No frames were captured, connection likely failed
            logger.error(f"Failed to start detection for device {device_id} due to connection issues")
            stop_events.pop(device_id)
            frame_queues.pop(device_id)
            return
        else:
            # Update device status to online only if frames were successfully captured
            update_device_status(device_id, "online")

            # Start the processing thread
            process_thread = threading.Thread(target=detect_and_process_frames, args=(device_id, update_callback))
            process_threads[device_id] = process_thread
            process_thread.start()

            active_cameras[device_id] = rtsp_url
            logger.info(f"Started detection for device {device_id}. Total active devices: {len(capture_threads)}")
    else:
        logger.error(f"Failed to start detection for device {device_id} due to connection issues")
        stop_events.pop(device_id)
        if device_id in frame_queues:
            frame_queues.pop(device_id)

def stop_detection(device_id):
    logger.info(f"Attempting to stop detection for device {device_id}")
    if device_id in stop_events:
        stop_events[device_id].set()

    if device_id in capture_threads:
        logger.debug(f"Stopping capture thread for device {device_id}")
        capture_threads[device_id].join(timeout=5)
        capture_threads.pop(device_id)
    else:
        logger.warning(f"No capture thread found for device {device_id}")

    if device_id in process_threads:
        logger.debug(f"Stopping process thread for device {device_id}")
        process_threads[device_id].join(timeout=5)
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

    update_device_status(device_id, "offline")
    logger.info(f"Stopped detection for device {device_id}. Total active devices: {len(capture_threads)}")

def list_active_cameras():
    return list(active_cameras.keys())

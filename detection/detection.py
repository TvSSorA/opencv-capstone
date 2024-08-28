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
from detection.db_interactions import save_basic_image_metadata, save_annotated_frame_metadata, update_device_status, get_rtsp_url
from detection.image_processing import ensure_directory_exists, save_images, create_annotated_frames, send_cropped_frame, send_annotated_and_heatmap

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


def capture_frames(rtsp_url, device_id):
    logger.info(f"Starting frame capture for device {device_id} with URL {rtsp_url}")
    retries = 0
    cap = None

    while retries < Config.MAX_RETRIES:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(
                f"Failed to open stream for device {device_id}, retrying... ({retries + 1}/{Config.MAX_RETRIES})")
            retries += 1
            time.sleep(Config.RETRY_DELAY)
            continue

        # Check if the camera is providing valid frames
        ret, frame = cap.read()
        if ret:
            logger.info(f"Successfully connected to device {device_id}.")
            # Update status to online only after successfully capturing a valid frame
            update_device_status(device_id, "online")
            break
        else:
            logger.error(
                f"Failed to read initial frame for device {device_id}, retrying... ({retries + 1}/{Config.MAX_RETRIES})")
            retries += 1
            time.sleep(Config.RETRY_DELAY)

    if retries >= Config.MAX_RETRIES or not cap.isOpened():
        logger.error(f"Max retries reached for device {device_id}. Connection failed.")
        if cap:
            cap.release()
        update_device_status(device_id, "offline")
        stop_events[device_id].set()  # Signal to stop detection
        return  # Early exit if connection failed

    # Proceed with frame capturing
    while cap.isOpened() and not stop_events[device_id].is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame for device {device_id}, retrying...")
            retries += 1
            time.sleep(Config.RETRY_DELAY)
            if retries >= Config.MAX_RETRIES:
                logger.error(f"Max retries reached for device {device_id}. Stopping capture.")
                update_device_status(device_id, "offline")  # Update status to offline if connection fails
                stop_events[device_id].set()  # Signal to stop detection
                break
            continue

        if frame_queues[device_id].full():
            continue
        frame_queues[device_id].put(frame)

    cap.release()
    logger.info(f"Stopped frame capture for device {device_id}")


def capture_frames(rtsp_url, device_id):
    logger.info(f"Starting frame capture for device {device_id} with URL {rtsp_url}")
    retries = 0
    cap = None

    while retries < Config.MAX_RETRIES:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(
                f"Failed to open stream for device {device_id}, retrying... ({retries + 1}/{Config.MAX_RETRIES})")
            retries += 1
            time.sleep(Config.RETRY_DELAY)
            continue

        # Check if the camera is providing valid frames
        ret, frame = cap.read()
        if ret:
            logger.info(f"Successfully connected to device {device_id}.")
            # Update status to online only after successfully capturing a valid frame
            update_device_status(device_id, "online")
            break
        else:
            logger.error(
                f"Failed to read initial frame for device {device_id}, retrying... ({retries + 1}/{Config.MAX_RETRIES})")
            retries += 1
            time.sleep(Config.RETRY_DELAY)

    if retries >= Config.MAX_RETRIES or not cap.isOpened():
        logger.error(f"Max retries reached for device {device_id}. Connection failed.")
        if cap:
            cap.release()
        update_device_status(device_id, "offline")
        stop_detection(device_id)  # Stop detection for the device
        return  # Early exit if connection failed

    # Proceed with frame capturing
    while cap.isOpened() and not stop_events[device_id].is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame for device {device_id}, retrying...")
            retries += 1
            time.sleep(Config.RETRY_DELAY)
            if retries >= Config.MAX_RETRIES:
                logger.error(f"Max retries reached for device {device_id}. Stopping capture.")
                update_device_status(device_id, "offline")  # Update status to offline if connection fails
                stop_detection(device_id)  # Stop detection for the device
                break
            continue

        if frame_queues[device_id].full():
            continue
        frame_queues[device_id].put(frame)

    cap.release()
    logger.info(f"Stopped frame capture for device {device_id}")


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
    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, device_id))
    process_thread = threading.Thread(target=detect_and_process_frames, args=(device_id, update_callback))

    capture_threads[device_id] = capture_thread
    process_threads[device_id] = process_thread

    capture_thread.start()
    process_thread.start()

    active_cameras[device_id] = rtsp_url
    logger.info(f"Started detection for device {device_id}. Total active devices: {len(capture_threads)}")

def stop_detection(device_id):
    logger.info(f"Attempting to stop detection for device {device_id}")
    if device_id in stop_events:
        stop_events[device_id].set()

    if device_id in capture_threads:
        if threading.current_thread() != capture_threads[device_id]:
            logger.debug(f"Stopping capture thread for device {device_id}")
            capture_threads[device_id].join(timeout=5)
        capture_threads.pop(device_id)

    if device_id in process_threads:
        logger.debug(f"Stopping process thread for device {device_id}")
        process_threads[device_id].join(timeout=5)
        process_threads.pop(device_id)

    if device_id in frame_queues:
        logger.debug(f"Clearing frame queue for device {device_id}")
        frame_queues.pop(device_id)

    if device_id in active_cameras:
        logger.debug(f"Removing active camera entry for device {device_id}")
        active_cameras.pop(device_id)

    if device_id in stop_events:
        logger.debug(f"Removing stop event for device {device_id}")
        stop_events.pop(device_id)

    update_device_status(device_id, "offline")
    logger.info(f"Stopped detection for device {device_id}. Total active devices: {len(capture_threads)}")

def list_active_cameras():
    return list(active_cameras.keys())

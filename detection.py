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
from pymongo import MongoClient
import base64
import asyncio

# Import utils
from utils import logger

# MongoDB setup
client = MongoClient('mongodb+srv://adam123:tntguy123@vnmc-database.r8b4uv0.mongodb.net/')
db = client['capstone-project']
collection = db['images']
devices_collection = db['devices']

# Configuration
class CFG:
    MODEL_WEIGHTS = 'weights/best.pt'  # yolov8s.pt, yolov9c.pt, yolov9e.pt
    CONFIDENCE = 0.35
    IOU = 0.5
    HEATMAP_ALPHA = 0.2
    RADIUS = 40
    TRACK_THRESH = 0.35
    TRACK_SECONDS = 5
    MATCH_THRESH = 0.9999
    FRAME_RATE = 20
    MAX_RETRIES = 5
    RETRY_DELAY = 5  # seconds to retry connection

# Initialize the YOLO model
model = YOLO(CFG.MODEL_WEIGHTS)
tracker = sv.ByteTrack(
    track_activation_threshold=CFG.TRACK_THRESH,
    lost_track_buffer=CFG.TRACK_SECONDS * CFG.FRAME_RATE,
    minimum_matching_threshold=CFG.MATCH_THRESH,
    frame_rate=CFG.FRAME_RATE
)

# Set up annotators
box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
heat_map_annotator = sv.HeatMapAnnotator(
    position=sv.Position.BOTTOM_CENTER,
    opacity=CFG.HEATMAP_ALPHA,
    radius=CFG.RADIUS,
    kernel_size=25,
    top_hue=0,
    low_hue=125
)

# Directories to save images
images_dir = 'outputs'
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

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

async def send_update(data, update_callback):
    await update_callback(data)

def process_frame(device_id, frame, results, update_callback=None):
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
    ensure_directory_exists(output_dir)
    ensure_directory_exists(annotated_output_dir)
    ensure_directory_exists(whole_frame_dir)
    ensure_directory_exists(heatmap_output_dir)

    for i in range(min_length):
        box = human_detections.xyxy.tolist()[i]
        uuid_label = labels[i]
        new_detected_uuids.add(uuid_label)
        if uuid_label not in saved_images_ids:
            x1, y1, x2, y2 = box
            crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
            now = datetime.now()
            timestamp = int(now.timestamp() * 1000)
            date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            cropped_frame_name = f"{uuid_label}-{date_time}.jpg"
            whole_frame_name = f"{uuid_label}-whole-{date_time}.jpg"
            cv2.imwrite(os.path.join(output_dir, cropped_frame_name), crop_object)
            cv2.imwrite(os.path.join(whole_frame_dir, whole_frame_name), frame)
            saved_images_ids.add(uuid_label)

            collection.insert_one({
                "_id": uuid_label,
                "device_id": device_id,
                "file_name": cropped_frame_name,
                "time": timestamp
            })

    if new_detected_uuids - current_uuids:
        current_uuids.update(new_detected_uuids)
    annotated_frame = box_annotator.annotate(frame.copy(), detections=human_detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=human_detections, labels=labels)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=human_detections)
    heatmap_frame = heat_map_annotator.annotate(frame.copy(), detections=human_detections)
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    cv2.imwrite(os.path.join(annotated_output_dir, 'annotated-' + date_time + '.jpg'), annotated_frame)
    cv2.imwrite(os.path.join(heatmap_output_dir, 'heatmap-' + date_time + '.jpg'), heatmap_frame)

    if update_callback:
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        data = {
            "image": encoded_frame,
            "device_id": device_id,
            "file_name": "test.jpg",
            "time": int(now.timestamp() * 1000)
        }
        asyncio.run(send_update(data, update_callback))
    return annotated_frame

def capture_frames(rtsp_url, device_id):
    logger.info(f"Starting frame capture for device {device_id} with URL {rtsp_url}")
    retries = 0
    while retries < CFG.MAX_RETRIES:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"Failed to open stream for device {device_id}, retrying... ({retries + 1}/{CFG.MAX_RETRIES})")
            retries += 1
            time.sleep(CFG.RETRY_DELAY)
            continue

        while cap.isOpened() and not stop_events[device_id].is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame for device {device_id}, retrying... ({retries + 1}/{CFG.MAX_RETRIES})")
                retries += 1
                time.sleep(CFG.RETRY_DELAY)
                break

            if frame_queues[device_id].full():
                continue
            frame_queues[device_id].put(frame)

        if retries >= CFG.MAX_RETRIES:
            logger.error(f"Max retries reached for device {device_id}. Stopping capture.")
            break

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
        if elapsed_time < 1 / CFG.FRAME_RATE:
            time.sleep(1 / CFG.FRAME_RATE - elapsed_time)

        frame = frame_queues[device_id].get()
        results = model.predict(source=frame, show=False, stream=True, classes=[0], imgsz=640)
        for result in results:
            annotated_frame = process_frame(device_id, frame, result, update_callback)
            if annotated_frame is not None:
                logger.info(f"New frame processed and saved for device {device_id}.")
        last_frame_time = current_time

    logger.info(f"Stopped frame processing for device {device_id}")

def get_rtsp_url(device_id):
    device = devices_collection.find_one({"_id": device_id})
    return device.get("rtsp_url") if device else None

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

    devices_collection.update_one({"_id": device_id}, {"$set": {"status": "online"}})

    capture_thread.start()
    process_thread.start()

    active_cameras[device_id] = rtsp_url
    logger.info(f"Started detection for device {device_id}. Total active devices: {len(capture_threads)}")

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

    devices_collection.update_one({"_id": device_id}, {"$set": {"status": "offline"}})
    logger.info(f"Stopped detection for device {device_id}. Total active devices: {len(capture_threads)}")

def list_active_cameras():
    return list(active_cameras.keys())

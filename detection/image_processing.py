# image_processing.py
import os
import cv2
import numpy as np
import supervision as sv
from datetime import datetime
from .utils import logger
import base64

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_images(frame, box, uuid_label, output_dir, whole_frame_dir, single_box_annotated_dir, human_detections):
    try:
        x1, y1, x2, y2 = box
        crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

        crop_path = os.path.join(output_dir, uuid_label + '-' + date_time + '.jpg')
        whole_frame_path = os.path.join(whole_frame_dir, uuid_label + '-whole-' + date_time + '.jpg')
        single_box_path = os.path.join(single_box_annotated_dir, uuid_label + '-single-box-' + date_time + '.jpg')

        cv2.imwrite(crop_path, crop_object)
        cv2.imwrite(whole_frame_path, frame)
        logger.info(f"Saved cropped image at {crop_path} and whole frame at {whole_frame_path}")

        single_box_frame = frame.copy()
        detection = sv.Detections(
            xyxy=np.array([box]),
            class_id=np.array([0]),
            confidence=np.array([human_detections.confidence[0]]),
            tracker_id=np.array([human_detections.tracker_id[0]])
        )
        single_box_frame = sv.LabelAnnotator().annotate(single_box_frame, detections=detection, labels=[uuid_label])
        cv2.rectangle(single_box_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite(single_box_path, single_box_frame)
        logger.info(f"Saved single box annotated image at {single_box_path}")

        return crop_path, whole_frame_path, single_box_path, date_time
    except Exception as e:
        logger.error(f"Error saving images for person {uuid_label}: {e}")
        return None, None, None, None

def create_annotated_frames(frame, human_detections, labels, annotated_output_dir, heatmap_output_dir, box_annotator, label_annotator, trace_annotator, heat_map_annotator):
    try:
        annotated_frame = box_annotator.annotate(frame.copy(), detections=human_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=human_detections, labels=labels)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections=human_detections)
        heatmap_frame = heat_map_annotator.annotate(frame.copy(), detections=human_detections)

        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        cv2.imwrite(os.path.join(annotated_output_dir, 'annotated-' + date_time + '.jpg'), annotated_frame)
        cv2.imwrite(os.path.join(heatmap_output_dir, 'heatmap-' + date_time + '.jpg'), heatmap_frame)
        logger.info(f"Saved annotated frame and heatmap frame for time {date_time}")

        return annotated_frame
    except Exception as e:
        logger.error(f"Error creating annotated frames: {e}")
        return None

async def send_update(data, update_callback):
    try:
        await update_callback(data)
        logger.info("Sent update successfully")
    except Exception as e:
        logger.error(f"Failed to send update: {e}")

async def send_update_to_clients(device_id, annotated_frame, uuid_label, crop_path, update_callback):
    try:
        # Encode annotated frame
        _, annotated_buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_annotated_frame = base64.b64encode(annotated_buffer).decode('utf-8')

        # Read and encode cropped frame
        crop_frame = cv2.imread(crop_path)
        _, crop_buffer = cv2.imencode('.jpg', crop_frame)
        encoded_crop_frame = base64.b64encode(crop_buffer).decode('utf-8')

        data = {
            "image": encoded_crop_frame,
            "annotated": encoded_annotated_frame,
            "device_id": device_id,
            "file_name": crop_path,
            "time": int(datetime.now().timestamp() * 1000)
        }
        await update_callback(data)
    except Exception as e:
        logger.error(f"Error sending update to clients for person {uuid_label}: {e}")

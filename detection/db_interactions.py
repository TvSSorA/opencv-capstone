import os
from pymongo import MongoClient
from datetime import datetime
from config import Config
from .utils import logger

client = MongoClient(Config.MONGO_URI)
db = client['capstone-project']
images_collection = db['images']
devices_collection = db['devices']

"""
def save_image_metadata(uuid_label, device_id, crop_path, whole_frame_path, single_box_path):
    try:
        crop_url = f"/static/{os.path.relpath(crop_path, 'static')}"
        whole_frame_url = f"/static/{os.path.relpath(whole_frame_path, 'static')}"
        single_box_url = f"/static/{os.path.relpath(single_box_path, 'static')}"

        images_collection.insert_one({
            'uuid': uuid_label,
            'device_id': device_id,
            'entry_timestamp': datetime.now().isoformat(),
            'exit_timestamp': None,
            'date': datetime.now().date().isoformat(),
            'crop_url': crop_url,
            'whole_frame_url': whole_frame_url,
            'single_box_url': single_box_url
        })
        logger.info(f"Saved URLs in MongoDB for person {uuid_label} with device ID {device_id}")
    except Exception as e:
        logger.error(f"Error saving URLs in MongoDB for person {uuid_label}: {e}")
"""

"""
def update_exit_timestamps(current_uuids, new_detected_uuids):
    try:
        for uuid_label in current_uuids - new_detected_uuids:
            images_collection.update_one(
                {'uuid': uuid_label},
                {'$set': {'exit_timestamp': datetime.now().isoformat()}}
            )
            logger.info(f"Updated exit timestamp for person {uuid_label}")
    except Exception as e:
        logger.error(f"Error updating exit timestamps: {e}")
"""

def save_basic_image_metadata(uuid_label, device_id, cropped_frame_name, timestamp):
    try:
        images_collection.insert_one({
            'uuid': uuid_label,
            'device_id': device_id,
            'file_name': cropped_frame_name,
            'time': timestamp
        })
        logger.info(f"Saved basic image metadata in MongoDB for person {uuid_label} with device ID {device_id}")
    except Exception as e:
        logger.error(f"Error saving basic image metadata in MongoDB for person {uuid_label}: {e}")

def update_device_status(device_id, status):
    try:
        devices_collection.update_one({"_id": device_id}, {"$set": {"status": status}})
        logger.info(f"Updated device status to {status} for device ID {device_id}")
    except Exception as e:
        logger.error(f"Error updating device status for device {device_id}: {e}")

def get_rtsp_url(device_id):
    try:
        device = devices_collection.find_one({"_id": device_id})
        rtsp_url = device.get("rtsp_url") if device else None
        if rtsp_url:
            logger.info(f"Retrieved RTSP URL for device {device_id}: {rtsp_url}")
        else:
            logger.warning(f"No RTSP URL found for device {device_id}")
        return rtsp_url
    except Exception as e:
        logger.error(f"Error retrieving RTSP URL for device {device_id}: {e}")
        return None

# app.py

from fastapi import FastAPI, WebSocket, HTTPException
from multiprocessing import Pool, Value
import os
from datetime import datetime
from pymongo import MongoClient
import threading
import json

# Import from detection subdirectory
from detection.detection import start_detection, stop_detection, list_active_cameras
from loguru import logger

app = FastAPI()

# MongoDB setup
client = MongoClient('mongodb+srv://adam123:tntguy123@vnmc-database.r8b4uv0.mongodb.net/')
db = client['capstone-project']
collection = db['images']
devices_collection = db['devices']

pool = None
is_running = {}


@app.on_event("startup")
async def startup_event():
    global pool
    pool = Pool(processes=1)
    logger.info("Application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    global pool
    if pool:
        pool.terminate()
        pool.join()
    logger.info("Application shutdown complete.")


@app.post("/start-detection/{device_id}")
async def start_detection_api(device_id: str):
    global is_running
    if device_id in is_running and is_running[device_id]:
        logger.warning(f"Detection already started for device {device_id}")
        return {"status": "detection already started"}

    is_running[device_id] = True
    start_detection(device_id, send_updates)
    logger.info(f"Detection started for device {device_id}")
    return {"status": "detection started"}


@app.post("/stop-detection/{device_id}")
async def stop_detection_api(device_id: str):
    global is_running
    if device_id not in is_running or not is_running[device_id]:
        logger.warning(f"Detection not running for device {device_id}")
        return {"status": "detection not running"}

    is_running[device_id] = False
    stop_detection(device_id)
    logger.info(f"Detection stopped for device {device_id}")
    return {"status": "detection stopped"}


@app.get("/list-active-cameras")
async def list_active_cameras_api():
    active_cameras = list_active_cameras()
    logger.info(f"Listing active cameras: {active_cameras}")
    return {"active_cameras": active_cameras}


@app.get("/latest-images")
async def get_latest_images():
    try:
        result = {}
        for device_id in list_active_cameras():
            current_date = datetime.now().strftime("%Y-%m-%d")
            whole_frame_dir = os.path.join("whole_frames", device_id, current_date)
            annotated_frame_dir = os.path.join("annotated_images", device_id, current_date)

            if os.path.exists(whole_frame_dir):
                whole_frame_files = sorted(os.listdir(whole_frame_dir),
                                           key=lambda x: os.path.getmtime(os.path.join(whole_frame_dir, x)),
                                           reverse=True)
                latest_whole_frame = whole_frame_files[0] if whole_frame_files else None
            else:
                latest_whole_frame = None

            if os.path.exists(annotated_frame_dir):
                annotated_frame_files = sorted(os.listdir(annotated_frame_dir),
                                               key=lambda x: os.path.getmtime(os.path.join(annotated_frame_dir, x)),
                                               reverse=True)
                latest_annotated_frame = annotated_frame_files[0] if annotated_frame_files else None
            else:
                latest_annotated_frame = None

            result[device_id] = {
                "latest_whole_frame": f"/whole_frames/{device_id}/{current_date}/{latest_whole_frame}" if latest_whole_frame else None,
                "latest_annotated_frame": f"/annotated_images/{device_id}/{current_date}/{latest_annotated_frame}" if latest_annotated_frame else None
            }

        return result
    except Exception as e:
        logger.error(f"Error getting latest images: {e}")
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while any(is_running.values()):
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connections.remove(websocket)
        await websocket.close()


connections = []


def send_updates(data):
    for connection in connections:
        try:
            connection.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending data: {e}")

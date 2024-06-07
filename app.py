from fastapi import FastAPI, WebSocket, HTTPException
from multiprocessing import Pool, Value
from detection.detection import start_detection, stop_detection
import os
from datetime import datetime
from pymongo import MongoClient
import threading
import json

app = FastAPI()

# MongoDB setup
client = MongoClient('mongodb+srv://adam123:tntguy123@vnmc-database.r8b4uv0.mongodb.net/')
db = client['capstone-project']
collection = db['images']
devices_collection = db['devices']

pool = None
is_running = Value('b', False)


@app.on_event("startup")
async def startup_event():
    global pool
    pool = Pool(processes=1)


@app.on_event("shutdown")
async def shutdown_event():
    global pool
    if pool:
        pool.terminate()
        pool.join()


@app.post("/start-detection/{device_id}")
async def start_detection_api(device_id: str):
    global pool, is_running
    if not pool:
        pool = Pool(processes=1)
    is_running.value = True
    pool.apply_async(start_detection, (device_id, send_updates))
    return {"status": "detection started"}


@app.post("/stop-detection/{device_id}")
async def stop_detection_api(device_id: str):
    global pool, is_running
    if pool:
        pool.terminate()
        pool.join()
        pool = None
    is_running.value = False
    stop_detection(device_id)
    return {"status": "detection stopped"}


@app.get("/latest-images")
async def get_latest_images():
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        whole_frame_dir = os.path.join("whole_frames", current_date)
        annotated_frame_dir = os.path.join("annotated_images", current_date)

        whole_frame_files = sorted(os.listdir(whole_frame_dir),
                                   key=lambda x: os.path.getmtime(os.path.join(whole_frame_dir, x)), reverse=True)
        annotated_frame_files = sorted(os.listdir(annotated_frame_dir),
                                       key=lambda x: os.path.getmtime(os.path.join(annotated_frame_dir, x)),
                                       reverse=True)

        latest_whole_frame = whole_frame_files[0] if whole_frame_files else None
        latest_annotated_frame = annotated_frame_files[0] if annotated_frame_files else None

        return {
            "latest_whole_frame": f"/whole_frames/{current_date}/{latest_whole_frame}" if latest_whole_frame else None,
            "latest_annotated_frame": f"/annotated_images/{current_date}/{latest_annotated_frame}" if latest_annotated_frame else None
        }
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while is_running.value:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connections.remove(websocket)
        await websocket.close()


connections = []


def send_updates(data):
    for connection in connections:
        try:
            connection.send_text(json.dumps(data))
        except Exception as e:
            print(f"Error sending data: {e}")

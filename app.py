from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from multiprocessing import Pool
from pymongo import MongoClient
from loguru import logger
import json
from detection.detection import start_detection, stop_detection, list_active_cameras

app = FastAPI()

allowed_origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
client = MongoClient('mongodb+srv://adam123:tntguy123@vnmc-database.r8b4uv0.mongodb.net/')
db = client['capstone-project']
collection = db['images']
devices_collection = db['devices']

pool = None
is_running = {}
connections = set()

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
    if device_id in is_running and is_running[device_id]:
        logger.warning(f"Detection already started for device {device_id}")
        raise HTTPException(status_code=400, detail={
            "status": "detection already started"
        })

    is_running[device_id] = True
    start_detection(device_id, send_updates)
    logger.info(f"Detection started for device {device_id}")
    return {"status": "detection started"}

@app.post("/stop-detection/{device_id}")
async def stop_detection_api(device_id: str):
    if device_id not in is_running or not is_running[device_id]:
        logger.warning(f"Detection not running for device {device_id}")
        raise HTTPException(status_code=400, detail={
            "status": "detection not running"
        })

    is_running[device_id] = False
    stop_detection(device_id)
    logger.info(f"Detection stopped for device {device_id}")
    return {"status": "detection stopped"}

@app.get("/list-active-cameras")
async def list_active_cameras_api():
    active_cameras = list_active_cameras()
    logger.info(f"Listing active cameras: {active_cameras}")
    return {"active_cameras": active_cameras}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connections.remove(websocket)

async def broadcast_active_cameras(websocket=None):
    active_cameras = list_active_cameras()
    message = json.dumps({"active_cameras": active_cameras})
    if websocket:
        await websocket.send_text(message)
    else:
        for connection in connections:
            await connection.send_text(message)

async def send_updates(data):
    for connection in connections:
        try:
            await connection.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            connections.remove(connection)

class Config:
    MODEL_WEIGHTS = 'weights/best.pt'
    CONFIDENCE = 0.35
    IOU = 0.5
    HEATMAP_ALPHA = 0.2
    RADIUS = 40
    TRACK_THRESH = 0.35
    TRACK_SECONDS = 5
    MATCH_THRESH = 0.9999
    FRAME_RATE = 20
    MAX_RETRIES = 2  #  max retries
    RETRY_DELAY = 2  # Delay in seconds between retry attempts
    MONGO_URI = 'mongodb+srv://adam123:tntguy123@vnmc-database.r8b4uv0.mongodb.net/'
    OUTPUT_DIR = 'outputs'

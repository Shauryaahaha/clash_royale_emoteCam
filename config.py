import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "./data/recordings")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/best_model.onnx")
    EMOTE_ASSETS = os.getenv("EMOTE_ASSETS", "./src/assets/emotes")
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    DEBOUNCE_TIME = float(os.getenv("DEBOUNCE_TIME", 0.6))
    DEBUG = os.getenv("DEBUG", "False").lower() in ("1","true","yes")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SAVE_DEBUG_VIDEOS = os.getenv("SAVE_DEBUG_VIDEOS", "False").lower() in ("1","true","yes")


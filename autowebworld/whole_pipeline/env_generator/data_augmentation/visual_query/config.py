import os

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = "https://newapi.deepwisdom.ai/v1"
MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 32768

IMAGE_RESIZE_WIDTH = 512
IMAGE_RESIZE_HEIGHT = 512
IMAGE_QUALITY = 90


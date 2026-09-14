import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

GEMINI_MAX_TOKENS = int(
    os.getenv(
        "GEMINI_MAX_TOKENS",
        "1024"
    )
)

OCR_ENGINE = os.getenv(
    "OCR_ENGINE",
    "easyocr"
)

OCR_LANGUAGE = os.getenv(
    "OCR_LANGUAGE",
    "en"
)
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_FILE_SIZE_MB = 10
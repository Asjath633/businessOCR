import os

from dotenv import load_dotenv

load_dotenv()


# --- Ollama Cloud LLM Settings ---

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-oss:120b",
)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://ollama.com",
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.0",
    )
)

LLM_MAX_TOKENS = int(
    os.getenv(
        "LLM_MAX_TOKENS",
        "1024",
    )
)


# --- OCR Settings ---

OCR_ENGINE = os.getenv(
    "OCR_ENGINE",
    "ppocrv6",
)

OCR_LANGUAGE = os.getenv(
    "OCR_LANGUAGE",
    "en",
)


# --- File Settings ---

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_FILE_SIZE_MB = 10
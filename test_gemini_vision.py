from google import genai
from google.genai import types
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.5-flash-lite"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=API_KEY)

image_path = Path("inputs/camera_capture.jpg")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

image_bytes = image_path.read_bytes()

response = client.models.generate_content(
    model=MODEL,
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        ),
        """
Look at this image.

Determine ONLY whether it is clearly a business card.

Return ONLY valid JSON:

{
  "is_business_card": true
}

or:

{
  "is_business_card": false
}

Do not extract any text.
Do not provide explanations.
""",
    ],
)

print("\n===== GEMINI RESPONSE =====")
print(response.text)
print("===========================\n")
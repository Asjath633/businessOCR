"""
Gemini API client.

Handles:
- Text-only LLM requests
- Vision LLM requests
"""

from pathlib import Path
import mimetypes

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_TOKENS,
)


class LLMClient:
    """Client wrapper for Google Gemini API."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set! "
                "Please add it to the .env file."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model = GEMINI_MODEL
        self.max_tokens = GEMINI_MAX_TOKENS

    def chat(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=self.max_tokens,
            ),
        )

        return response.text

    def chat_with_image(
        self,
        system_prompt: str,
        user_message: str,
        image_path: str,
    ) -> str:

        image_file = Path(image_path)

        if not image_file.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image_bytes = image_file.read_bytes()

        mime_type = (
            mimetypes.guess_type(image_file.name)[0]
            or "image/jpeg"
        )

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                image_part,
                user_message,
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=self.max_tokens,
            ),
        )

        return response.text
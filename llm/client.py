"""
Ollama Cloud API client.

Handles:
- Text-only LLM requests
- Structured JSON responses

GPT-OSS is text-only, so image/vision requests are not supported here.
PP-OCRv6 handles image -> text before this client is called.
"""

import os
import ollama

from config import (
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


class LLMClient:
    """Client wrapper for Ollama Cloud API."""

    def __init__(self):

        api_key = os.getenv("OLLAMA_API_KEY")

        if not api_key:
            raise ValueError(
                "OLLAMA_API_KEY is not set! "
                "Please add it to the .env file."
            )

        self.client = ollama.Client(
            host=LLM_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}"
            },
        )

        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        response_format=None,
    ) -> str:
        """
        Send a text request to GPT-OSS.

        response_format:
            Optional JSON schema used to force structured output.
        """

        kwargs = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],

            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        # Force structured JSON when a schema is provided
        if response_format is not None:
            kwargs["format"] = response_format

        response = self.client.chat(**kwargs)

        return response.message.content
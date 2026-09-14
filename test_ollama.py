import os
from dotenv import load_dotenv
import ollama

load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

if not api_key:
    raise ValueError("OLLAMA_API_KEY is missing from .env")

client = ollama.Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {api_key}"
    }
)

response = client.chat(
    model="gpt-oss:120b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: OK"
        }
    ],
)

print("GPT-OSS RESPONSE:")
print(response.message.content)
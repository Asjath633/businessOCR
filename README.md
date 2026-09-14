# Business Card OCR (Terminal CLI)

Extract structured contact information from business card images using OCR + LLM directly from your terminal.

## Features

- **OCR Extraction** — Tesseract or EasyOCR for raw text extraction
- **LLM Structuring** — Groq / OpenRouter / Llama 3.3 70B to parse raw text into clean structured JSON
- **Terminal CLI Interface** — Run directly with image paths or auto-indexed numbers
- **Validation** — Pydantic schemas for type-safe, validated output

## Project Structure

```
business_card/
├── main.py                 # Terminal CLI entry point
├── ocr/
│   ├── __init__.py
│   └── extractor.py        # OCR logic (Tesseract / EasyOCR)
├── llm/
│   ├── __init__.py
│   ├── client.py           # LLM API client
│   └── extractor.py        # OCR text → structured JSON
├── schemas/
│   └── business_card.py    # Pydantic models
├── utils/
│   └── helpers.py          # Utility functions
├── config.py               # Configuration (env vars)
├── requirements.txt
├── .env                    # API keys (never commit)
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Edit [.env](file:///e:/API_OCR/business_card/.env) and set your API key and provider URL:

#### For Groq (Recommended):
```env
LLM_API_KEY=gsk_your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

#### For OpenRouter:
```env
LLM_API_KEY=sk-or-v1-your-openrouter-key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.3-70b-instruct
```

### 3. Usage in Terminal

#### List all available images:
```bash
python main.py list
```

#### Run latest image automatically:
```bash
python main.py
```

#### Run by image index number:
```bash
python main.py 1
```

#### Run by specific image path:
```bash
python main.py inputs/Camera/IMG_20200209_184151.jpg
```

## JSON Output Structure

```json
{
  "name": "Sushilkumar Shinde",
  "job_title": "ASSO. FINANCIAL SERVICES MGR",
  "company": "ICICI Prudential Life Insurance Co. Ltd",
  "email": "sushilkumar.shinde@iciciprulife.com",
  "phone": [
    "8412902672"
  ],
  "website": "www.iciciprulife.com",
  "address": "3rd Floor, R Square, Plot No: 62, College Road, Nashik 422005",
  "linkedin": null,
  "other": {}
}
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | Groq or OpenRouter API key |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Provider API endpoint URL |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model ID to use for extraction |
| `OCR_ENGINE` | `tesseract` / `easyocr` | OCR engine choice |
| `OCR_LANGUAGE` | `eng` | OCR language code |

## License

MIT


# Business Card OCR (Terminal CLI)

Extract structured contact information from business card images using OCR + LLM directly from your terminal.

## Features

- **OCR Extraction** — paddle-ocr for raw text extraction
- **LLM Structuring** — gpt-oss:120b to parse raw text into clean structured JSON
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
LLM_API_KEY=ollama API key
LLM_BASE_URL=https://ollama.com
LLM_MODEL=gpt-oss:120b
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
| `LLM_API_KEY` | — | ollama API key |
| `LLM_BASE_URL` | ' | 'https://ollama.com' Provider API endpoint URL |
| `LLM_MODEL` | `gpt-oss:120b` | Model ID to use for extraction |
| `OCR_ENGINE` | 'ppocrv6' | OCR engine choice |
| `OCR_LANGUAGE` | `eng` | OCR language code |

## License

MIT


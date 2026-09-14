"""
LLM-based business-card detector and structured data extractor.

Pipeline:

    1. PP-OCRv6 extracts text from the business-card image.
    2. OCR text is sent to GPT-OSS 120B Cloud.
    3. GPT-OSS checks whether the text is from a business card.
    4. GPT-OSS extracts structured fields.
    5. Pydantic validates the final result.

The image is NOT sent to the LLM.
"""

from llm.client import LLMClient
from schemas.business_card import BusinessCardData


SYSTEM_PROMPT = """
You are a business card data extraction assistant.

The application provides raw OCR text extracted from an image.

Your task has TWO responsibilities:

1. Determine whether the OCR text appears to come from a business card.
2. If it is a business card, extract the available business-card information.

The response must follow the provided JSON schema exactly.

BUSINESS CARD DETECTION:

Return is_business_card=true when the OCR text clearly contains
professional business-card information.

A business card normally contains some combination of:

- person's name
- company or organization
- designation
- phone number
- email
- website
- business address

Return is_business_card=false for text that appears to come from:

- receipts
- invoices
- forms
- certificates
- ID cards
- passports
- menus
- posters
- advertisements
- product labels
- screenshots
- ordinary documents
- letters
- webpages
- social-media screenshots
- other non-business-card content

If is_business_card is false:

- name = []
- organization = null
- designation = null
- email = []
- phone = []
- address = null

STRICT EXTRACTION RULES:

1. Use ONLY information explicitly present in the OCR text.
2. NEVER invent information.
3. NEVER hallucinate missing information.
4. NEVER autocomplete email addresses.
5. NEVER autocomplete phone numbers.
6. NEVER infer an email domain from the company name.
7. NEVER infer phone numbers from context.
8. NEVER use external knowledge.
9. OCR text may contain recognition errors.
10. Correct an OCR error only when the intended value is clearly
    supported by the OCR text itself.
11. If an email is incomplete or unclear, return [].
12. If a phone number is incomplete or unclear, return [].
13. If a name is unclear, do not guess.
14. If a company name is unclear, preserve the OCR text.
15. If a designation is unclear, preserve the OCR text or return null.
16. If an address is incomplete, preserve only the information present.
17. Preserve the original OCR information as faithfully as possible.
18. When uncertain, prefer missing data over a guessed value.

IMPORTANT:

The OCR output is the source of truth.

Your job is STRUCTURING the OCR information, not performing
creative OCR correction.
"""


def extract_structured_data(raw_text: str) -> BusinessCardData:
    """
    Check whether OCR text belongs to a business card and extract
    structured information using ONE GPT-OSS 120B Cloud API call.
    """

    client = LLMClient()

    user_message = f"""
Analyze the following PP-OCRv6 text.

Determine whether it represents a business card and extract
the available business-card information.

Use ONLY the OCR text.

Do NOT guess missing information.
Do NOT autocomplete email addresses.
Do NOT autocomplete phone numbers.
Do NOT use external knowledge.

OCR TEXT:

--- OCR TEXT START ---

{raw_text}

--- OCR TEXT END ---
"""

    # ---------------------------------------------------------
    # ONE GPT-OSS API CALL
    # ---------------------------------------------------------

    response = client.chat(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        response_format=BusinessCardData.model_json_schema(),
    )

    # ---------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------

    response = response.strip()

    try:
        data_dict = __import__("json").loads(response)

    except Exception as e:
        raise ValueError(
            f"GPT-OSS returned invalid JSON:\n{response}"
        ) from e

    # ---------------------------------------------------------
    # Pydantic validation
    # ---------------------------------------------------------

    return BusinessCardData(**data_dict)
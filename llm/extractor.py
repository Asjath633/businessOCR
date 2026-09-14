"""
LLM-based structured data extractor.

Pipeline:
    1. PP-OCRv6 extracts text from the business-card image.
    2. OCR text is sent to the Gemini LLM.
    3. Gemini structures the OCR text into JSON.
    4. Pydantic validates the final result.

The image is NOT sent to the LLM.
"""

import json
import re

from llm.client import LLMClient
from schemas.business_card import BusinessCardData


SYSTEM_PROMPT = """
You are a business card data extraction assistant.

The application provides raw OCR text extracted from a business card.

Your task is to convert ONLY the information present in the OCR text
into a structured JSON object.

FIELDS:

- name: List of full name(s) found on the card
- organization: Company or organization name
- designation: Job title or designation
- email: List of all email addresses found on the card
- phone: List of all phone numbers found on the card
- address: Full physical address, or null if not present

STRICT RULES:

1. Return ONLY valid JSON.
2. Do not return markdown, explanations, comments, or extra text.
3. Use ONLY information explicitly present in the OCR text.
4. NEVER invent, hallucinate, autocomplete, or assume information.
5. NEVER create missing characters in an email address.
6. NEVER create missing digits in a phone number.
7. NEVER infer an email domain from the company name.
8. NEVER infer a phone number from context or external knowledge.
9. NEVER use your own knowledge about a person or company.
10. OCR text may contain recognition errors.
11. Correct an OCR error ONLY when the intended value is clearly
    supported by the text itself.
12. If an email is incomplete, unclear, or unreliable, return [].
13. If a phone number is incomplete, unclear, or unreliable, return [].
14. If a name is unclear, do not guess the missing characters.
15. If a company name is unclear, preserve the OCR text rather than
    inventing a better-known company name.
16. If a designation is unclear, preserve the OCR text or return null.
17. If an address is incomplete, preserve only the information actually
    present in the OCR text.
18. name, email, and phone MUST always be JSON arrays.
19. organization, designation, and address MUST be strings or null.
20. Do not add fields that are not defined in the requested schema.
21. Preserve the original OCR information as faithfully as possible.
22. When uncertain, prefer missing data over a guessed value.

IMPORTANT:

The OCR output is the source of truth.

Your job is STRUCTURING, not OCR correction.

If the OCR says something that is unclear, do NOT use your knowledge
to determine what it probably should be.

For example:

OCR:
    arunkumar@gudyonix.com

Return:
    "email": ["arunkumar@gudyonix.com"]

But if OCR says:
    unkumar gudyonix.com

Return:
    "email": []

Do NOT convert it into:
    "arunkumar@gudyonix.com"

Similarly, if OCR says:
    -91 B9250 49937

Do NOT convert it into a complete phone number.

Return [] unless the phone number can be reliably determined
from the OCR text itself.
"""


def extract_structured_data(raw_text: str) -> BusinessCardData:
    """
    Extract structured business-card information from PP-OCRv6 text.

    Args:
        raw_text: Raw text produced by PP-OCRv6.

    Returns:
        Validated BusinessCardData instance.
    """

    client = LLMClient()

    user_message = f"""
Extract the business-card information from the following PP-OCRv6 text.

IMPORTANT:
- Use ONLY the OCR text.
- Do NOT guess missing information.
- Do NOT autocomplete email addresses.
- Do NOT autocomplete phone numbers.
- Do NOT use external knowledge.
- If a value is unclear or incomplete, return null or [] as required
  by the schema.

--- OCR TEXT START ---
{raw_text}
--- OCR TEXT END ---

Return ONLY the JSON object.
"""

    response = client.chat(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
    )

    # Parse JSON returned by the LLM
    try:
        data_dict = json.loads(response)

    except json.JSONDecodeError:

        # Handle markdown code block if the model returns one
        json_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            re.DOTALL,
        )

        if not json_match:
            raise ValueError(
                f"LLM returned invalid JSON:\n{response}"
            )

        try:
            data_dict = json.loads(json_match.group(1))

        except json.JSONDecodeError:
            raise ValueError(
                f"LLM returned invalid JSON:\n{response}"
            )

    # Validate against Pydantic schema
    return BusinessCardData(**data_dict)
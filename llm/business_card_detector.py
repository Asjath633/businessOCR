"""
Business card text detector.

Pipeline:

    Image
      ↓
    PP-OCRv6
      ↓
    OCR Text
      ↓
    GPT-OSS 120B
      ↓
    Business card / Not business card

This module only checks whether OCR text
appears to come from a business card.
"""

import json
import re

from pydantic import BaseModel

from llm.client import LLMClient


# --------------------------------------------------
# Pydantic response schema
# --------------------------------------------------

class BusinessCardCheck(BaseModel):
    is_business_card: bool


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a business-card text classification assistant.

Your ONLY task is to determine whether the supplied OCR text
appears to come from a business card.

Return ONLY valid JSON in exactly this format:

{
  "is_business_card": true
}

or

{
  "is_business_card": false
}

RULES:

1. Use ONLY the supplied OCR text.

2. Return true ONLY when the text clearly appears to contain
   professional business-card information.

3. A business card normally contains some combination of:
   - person's name
   - company or organization
   - designation
   - phone number
   - email
   - website
   - business address

4. Return false for text that appears to come from:
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

5. Do NOT extract any information.

6. Do NOT return the person's name.

7. Do NOT return the company name.

8. Do NOT return OCR text.

9. Do NOT provide explanations.

10. Return ONLY the JSON object.

11. If you are not confident that the OCR text represents
    a business card, return false.

12. Do NOT use external knowledge.
"""


# --------------------------------------------------
# JSON parser
# --------------------------------------------------

def _parse_response(response: str) -> BusinessCardCheck:

    response = response.strip()

    # Normal JSON
    try:
        data = json.loads(response)
        return BusinessCardCheck(**data)

    except json.JSONDecodeError:
        pass

    # JSON inside markdown code block
    json_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        response,
        re.DOTALL,
    )

    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return BusinessCardCheck(**data)

        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    raise ValueError(
        "GPT-OSS returned invalid business-card classification:\n"
        + response
    )


# --------------------------------------------------
# Main detector function
# --------------------------------------------------

def is_business_card_text(raw_text: str) -> bool:

    client = LLMClient()

    user_message = f"""
Determine whether the following OCR text is from a business card.

--- OCR TEXT START ---
{raw_text}
--- OCR TEXT END ---

Return ONLY:

{{
  "is_business_card": true
}}

or:

{{
  "is_business_card": false
}}
"""

    response = client.chat(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
    )

    result = _parse_response(response)

    return result.is_business_card
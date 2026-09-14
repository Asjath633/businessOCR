"""
Business card image detector.

Pipeline:
    Image
      ↓
    Qwen3-VL Vision LLM
      ↓
    Business card / Not business card

This module ONLY checks whether the input image is
clearly a business card.

It does NOT perform OCR.
It does NOT extract contact information.
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
You are an image classification assistant.

Your ONLY task is to determine whether the provided image
is clearly a business card.

Return ONLY valid JSON in exactly this format:

{
  "is_business_card": true
}

or

{
  "is_business_card": false
}

RULES:

1. Analyze the image itself.
2. Return true ONLY when the image is clearly a business card
   or professional contact card.
3. Return false for:
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
   - photographs
   - webpages
   - social-media screenshots
   - other non-business-card images
4. A business card normally contains professional contact
   information such as a person's name, company, designation,
   phone number, email, website, or business address.
5. Do NOT extract any information from the card.
6. Do NOT return the person's name.
7. Do NOT return the company name.
8. Do NOT return OCR text.
9. Do NOT provide explanations.
10. Return ONLY the JSON object.
11. If you are not confident that the image is a business card,
    return false.
"""


# --------------------------------------------------
# JSON parser
# --------------------------------------------------

def _parse_response(response: str) -> BusinessCardCheck:
    """
    Parse the LLM response and validate it with Pydantic.
    """

    response = response.strip()

    # ----------------------------------------------
    # Try normal JSON first
    # ----------------------------------------------

    try:
        data = json.loads(response)
        return BusinessCardCheck(**data)

    except json.JSONDecodeError:
        pass

    # ----------------------------------------------
    # Try JSON inside markdown code block
    # ----------------------------------------------

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

    # ----------------------------------------------
    # If response is invalid
    # ----------------------------------------------

    raise ValueError(
        f"Qwen3-VL returned invalid business-card "
        f"classification:\n{response}"
    )


# --------------------------------------------------
# Main detector function
# --------------------------------------------------

def is_business_card_image(image_path: str) -> bool:
    """
    Determine whether an image is a business card.

    Args:
        image_path: Path to the image.

    Returns:
        True  -> image is a business card.
        False -> image is not a business card.
    """

    client = LLMClient()

    user_message = """
Look at the provided image.

Determine whether this image is clearly a business card.

Return ONLY:

{
  "is_business_card": true
}

or:

{
  "is_business_card": false
}
"""

    response = client.chat_with_image(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        image_path=image_path,
    )

    result = _parse_response(response)

    return result.is_business_card
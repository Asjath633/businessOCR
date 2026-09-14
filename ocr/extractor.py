from dataclasses import dataclass
from typing import List
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


@dataclass
class OCRResult:
    raw_text: str
    confidence: float
    best_preprocessing: str = "Full + Contact Region"


print("🔄 Loading PP-OCRv6...")

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("✓ PP-OCRv6 loaded.")


def get_ocr_data(result):
    """
    Extract OCR data from a PaddleOCR 3.x result.
    """

    if hasattr(result, "json"):
        data = result.json

        if isinstance(data, str):
            import json
            data = json.loads(data)

    elif isinstance(result, dict):
        data = result

    else:
        raise RuntimeError(
            f"Unsupported PaddleOCR result type: {type(result)}"
        )

    if isinstance(data, dict) and "res" in data:
        data = data["res"]

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected PaddleOCR result structure: {type(data)}"
        )

    return data


def run_ocr(image_path: str):
    """
    Run PP-OCRv6 on an image.
    """

    results = ocr.predict(image_path)

    texts: List[str] = []
    scores: List[float] = []

    for result in results:

        data = get_ocr_data(result)

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        for text in rec_texts:
            text = str(text).strip()

            if text:
                texts.append(text)

        for score in rec_scores:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass

    confidence = (
        (sum(scores) / len(scores)) * 100
        if scores
        else 0.0
    )

    return texts, scores, confidence


def create_contact_crop(image_path: str) -> str:
    """
    Crop the contact-information area of the business card.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    # Contact section
    x1 = int(width * 0.25)
    x2 = int(width * 0.53)

    y1 = int(height * 0.52)
    y2 = int(height * 0.98)

    crop = image[y1:y2, x1:x2]

    # Upscale 3x
    crop = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    # Keep image below PP-OCRv6 max side limit
    max_side = 3900

    crop_height, crop_width = crop.shape[:2]

    if max(crop_height, crop_width) > max_side:
        scale = max_side / max(crop_height, crop_width)

        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    # Sharpen
    blurred = cv2.GaussianBlur(
        crop,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        crop,
        1.5,
        blurred,
        -0.5,
        0
    )

    output_path = "inputs/contact_region.jpg"

    cv2.imwrite(
        output_path,
        sharpened
    )

    return output_path


def extract_text_from_image(image_path: str) -> OCRResult:
    """
    Extract business-card text using:

    1. Full-card PP-OCRv6
    2. Dedicated contact-region PP-OCRv6
    """

    print("\n🔍 Running full-card OCR...")

    full_texts, full_scores, full_confidence = run_ocr(
        image_path
    )

    print(
        f"✓ Full-card confidence: "
        f"{full_confidence:.2f}%"
    )

    print("\n--- FULL CARD OCR ---")

    for text in full_texts:
        print(text)

    # --------------------------------------------------
    # Contact region
    # --------------------------------------------------

    print("\n📱 Processing contact-information region...")

    try:

        contact_path = create_contact_crop(
            image_path
        )

        contact_texts, contact_scores, contact_confidence = run_ocr(
            contact_path
        )

        print(
            f"✓ Contact-region confidence: "
            f"{contact_confidence:.2f}%"
        )

        print("\n--- CONTACT REGION OCR ---")

        for text in contact_texts:
            print(text)

    except Exception as e:

        print(
            f"⚠️ Contact-region OCR failed: {e}"
        )

        contact_texts = []
        contact_scores = []
        contact_confidence = 0.0

    # --------------------------------------------------
    # Combine results
    # --------------------------------------------------

    combined_texts = []

    for text in full_texts + contact_texts:

        normalized = text.strip().lower()

        # Avoid exact duplicate lines
        duplicate = False

        for existing in combined_texts:

            if existing.strip().lower() == normalized:
                duplicate = True
                break

        if not duplicate:
            combined_texts.append(text)

    raw_text = "\n".join(combined_texts)

    # Combine confidence scores
    all_scores = (
        full_scores +
        contact_scores
    )

    confidence = (
        (sum(all_scores) / len(all_scores)) * 100
        if all_scores
        else 0.0
    )

    print("\n--- COMBINED OCR ---")
    print(raw_text)

    print("\n🏆 OCR processing completed.")

    return OCRResult(
        raw_text=raw_text,
        confidence=confidence,
        best_preprocessing="Full + Contact Region",
    )
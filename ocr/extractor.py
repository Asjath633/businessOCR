"""
PP-OCRv6 based text extraction.

Pipeline:

    Image
      ↓
    PP-OCRv6 (CPU)
      ↓
    Raw OCR text
      ↓
    GPT-OSS 120B Cloud
      ↓
    Structured JSON

Only ONE OCR inference is performed.

No contact-region OCR.
No extra crop.
No duplicate OCR pass.
"""
import cv2
from dataclasses import dataclass
from typing import List

from paddleocr import PaddleOCR


# ============================================================
# OCR RESULT
# ============================================================

@dataclass
class OCRResult:
    raw_text: str
    confidence: float
    best_preprocessing: str = "Full Card"


# ============================================================
# LOAD PP-OCRv6 ONCE
# ============================================================

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


# ============================================================
# PADDLEOCR RESULT PARSER
# ============================================================

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

    # PaddleOCR 3.x may return:
    #
    # {
    #     "res": {...}
    # }

    if isinstance(data, dict) and "res" in data:
        data = data["res"]

    if not isinstance(data, dict):

        raise RuntimeError(
            f"Unexpected PaddleOCR result structure: {type(data)}"
        )

    return data


# ============================================================
# SINGLE OCR PASS
# ============================================================

def run_ocr(image_path: str):
    """
    Run PP-OCRv6 on a resized image.

    Large camera images are resized before OCR to reduce
    CPU inference time while preserving enough resolution
    for business-card text.
    """

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    original_height, original_width = image.shape[:2]

    # --------------------------------------------------
    # Resize large images
    # --------------------------------------------------

    max_side = 1600

    current_max_side = max(
        original_width,
        original_height
    )

    if current_max_side > max_side:

        scale = max_side / current_max_side

        new_width = int(
            original_width * scale
        )

        new_height = int(
            original_height * scale
        )

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        print(
            f"📐 OCR resize: "
            f"{original_width}x{original_height} "
            f"→ "
            f"{new_width}x{new_height}"
        )

    else:

        print(
            f"📐 OCR image size: "
            f"{original_width}x{original_height}"
        )

    # --------------------------------------------------
    # ONE PP-OCRv6 inference
    # --------------------------------------------------

    results = ocr.predict(image)

    texts: List[str] = []
    scores: List[float] = []

    for result in results:

        data = get_ocr_data(result)

        rec_texts = data.get(
            "rec_texts",
            []
        )

        rec_scores = data.get(
            "rec_scores",
            []
        )

        for text in rec_texts:

            text = str(text).strip()

            if text:
                texts.append(text)

        for score in rec_scores:

            try:
                scores.append(
                    float(score)
                )

            except (TypeError, ValueError):
                pass

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    confidence = (
        (sum(scores) / len(scores)) * 100
        if scores
        else 0.0
    )

    return texts, scores, confidence


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def extract_text_from_image(
    image_path: str
) -> OCRResult:
    """
    Extract text from a business-card image.

    IMPORTANT:

    Only ONE PP-OCRv6 inference is performed.

    The entire card is processed directly.
    GPT-OSS will handle business-card detection
    and structured extraction afterward.
    """

    print("\n🔍 Running full-card OCR...")

    # ========================================================
    # ONE OCR CALL ONLY
    # ========================================================

    full_texts, full_scores, confidence = run_ocr(
        image_path
    )

    print(
        f"✓ Full-card confidence: "
        f"{confidence:.2f}%"
    )

    # ========================================================
    # RAW OCR
    # ========================================================

    print("\n--- FULL CARD OCR ---")

    for text in full_texts:
        print(text)

    # ========================================================
    # BUILD RAW TEXT
    # ========================================================

    raw_text = "\n".join(
        full_texts
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n🏆 OCR processing completed.")

    return OCRResult(
        raw_text=raw_text,
        confidence=confidence,
        best_preprocessing="Full Card",
    )
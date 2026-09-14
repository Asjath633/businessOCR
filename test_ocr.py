from paddleocr import PaddleOCR

IMAGE_PATH = "inputs/Camera/Klyonix (2).jpeg"

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
print("\n🔍 Running OCR...\n")

results = ocr.predict(IMAGE_PATH)

for result in results:
    result.print()
"""
CLI runner for Business Card OCR.
Run directly from terminal to extract structured contact data from an image file.

Usage:
    python main.py path/to/business_card.png
"""
import time
import warnings
warnings.filterwarnings("ignore")

import sys
import argparse
from ocr.extractor import extract_text_from_image
from llm.extractor import extract_structured_data


def process_business_card(image_path):

    print("=" * 50)
    print(f"📷 Processing: {image_path}")
    print("=" * 50)

    # --------------------------------------------------
    # STEP 1: OCR
    # --------------------------------------------------

    print("\n[1/2] 🔍 Extracting text using PP-OCRv6...")

    try:
        ocr_start = time.perf_counter()

        ocr_result = extract_text_from_image(
            image_path
        )

        ocr_time = time.perf_counter() - ocr_start
        print(f"⏱️ OCR time: {ocr_time:.2f} seconds")

    except Exception as e:

        print("\n❌ OCR failed:")
        print(e)
        print("=" * 50)
        return

    print(
        f"✓ Best OCR preprocessing: "
        f"{getattr(ocr_result, 'best_preprocessing', 'Full Card')}"
    )

    print(
        f"✓ OCR confidence: "
        f"{ocr_result.confidence:.2f}%"
    )

    print("\n--- RAW OCR TEXT ---")
    print(ocr_result.raw_text)

    # --------------------------------------------------
    # STEP 2: GPT-OSS
    # --------------------------------------------------

    print("\n[2/2] 🧠 Checking + structuring with GPT-OSS 120B...")

    try:
        llm_start = time.perf_counter()

        structured_data = extract_structured_data(
            ocr_result.raw_text
        )

        llm_time = time.perf_counter() - llm_start
        print(f"⏱️ GPT-OSS time: {llm_time:.2f} seconds")

    except Exception as e:
        print("\n❌ GPT-OSS failed:") 
        print(e)
        print("=" * 50)
        return

    # --------------------------------------------------
    # BUSINESS CARD CHECK
    # --------------------------------------------------

    if not structured_data.is_business_card:

        print("\n❌ This image is not a business card.")
        print("⏹ Processing stopped.")
        print("=" * 50)

        return

    print("\n✓ Business card detected.")

    # --------------------------------------------------
    # FINAL JSON
    # --------------------------------------------------

    print("\n✓ Pydantic validation completed.")

    print("\n--- STRUCTURED JSON ---")

    print(
        structured_data.model_dump_json(
            indent=2
        )
    )

    print("=" * 50)
import os
import sys

# Configure UTF-8 encoding for Windows console to handle emojis
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_all_images():
    """Find all image files in input folders, sorted newest first."""
    search_dirs = ["inputs", "Inout", "input", "inout"]
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    found = []
    seen = set()

    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        fp = os.path.normpath(os.path.join(root, f))
                        if fp not in seen:
                            seen.add(fp)
                            found.append((os.path.getmtime(fp), fp))

    # Sort newest first
    found.sort(key=lambda x: x[0], reverse=True)
    return [fp for _, fp in found]


def print_image_list(images):
    """Print numbered list of available images."""
    print("\n📋 Available Images in Folder:")
    print("-" * 50)
    for idx, img in enumerate(images, 1):
        latest_tag = " (Latest)" if idx == 1 else ""
        print(f"  [{idx}] {img}{latest_tag}")
    print("-" * 50)
    print("💡 To run an image by number, use: python main.py <number>")
    print("   Example: python main.py 2\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured data from a business card image.")
    parser.add_argument("arg", nargs="?", help="Image number (e.g. 1, 2), file path, 'camera', or 'list'")

    args = parser.parse_args()
    user_arg = args.arg

    # Handle 'camera' command for live webcam capture
    if user_arg and user_arg.lower() in ["camera", "cam"]:
        try:
            from inputs.capture import capture_image
        except ImportError:
            try:
                from camera.capture import capture_image
            except ImportError:
                print("❌ Could not import capture_image from inputs.capture or camera.capture")
                sys.exit(1)

        captured_path = capture_image()
        if captured_path:
            process_business_card(captured_path)
        sys.exit(0)

    images = get_all_images()

    if not images:
        print("❌ No images found in inputs/ or Inout/ folder!")
        print("💡 Place your business card images inside 'inputs' or 'Inout' folder or use 'python main.py camera'.")
        sys.exit(1)

    # Handle 'list' or '--list' command
    if user_arg in ["list", "-l", "--list"]:
        print_image_list(images)
        sys.exit(0)

    image_path = None

    if not user_arg:
        # Default to latest image (#1)
        image_path = images[0]
        print(f"ℹ️ No image specified. Auto-selected latest image [#1]: {image_path}")
        print("💡 Run 'python main.py list' to see all numbered images or 'python main.py camera' to capture live.")
    elif user_arg.isdigit():
        num = int(user_arg)
        if 1 <= num <= len(images):
            image_path = images[num - 1]
            print(f"ℹ️ Selected image #{num}: {image_path}")
        else:
            print(f"❌ Invalid image number: {num}. Must be between 1 and {len(images)}.")
            print_image_list(images)
            sys.exit(1)
    else:
        # Check if user_arg is a direct path or partial filename match
        if os.path.exists(user_arg):
            image_path = user_arg
        else:
            # Search for matching filename
            matches = [img for img in images if user_arg.lower() in img.lower()]
            if len(matches) == 1:
                image_path = matches[0]
                print(f"ℹ️ Matched image: {image_path}")
            elif len(matches) > 1:
                print(f"⚠️ Multiple images match '{user_arg}':")
                for m in matches:
                    print(f"  - {m}")
                print("\nPlease be more specific or use the image number.")
                sys.exit(1)
            else:
                print(f"❌ File not found: '{user_arg}'")
                print_image_list(images)
                sys.exit(1)

    process_business_card(image_path)



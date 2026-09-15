"""
CLI runner for Business Card OCR.
Run directly from terminal to extract structured contact data from an image file.

Usage:
    python main.py path/to/business_card.png
"""

import os
import sys
import psutil
import time
import warnings
import argparse

warnings.filterwarnings("ignore")

from ocr.extractor import extract_text_from_image
from llm.extractor import extract_structured_data


def process_business_card(image_path):

    # ==================================================
    # RESOURCE MONITORING - START
    # ==================================================

    process = psutil.Process(os.getpid())

    start_time = time.perf_counter()

    start_ram = process.memory_info().rss / (1024 * 1024)

    # Get input image size
    try:
        image_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    except Exception:
        image_size_mb = 0.0

    # ==================================================
    # START
    # ==================================================

    print("=" * 50)
    print(f"📷 Processing: {image_path}")
    print(f"📦 Image size: {image_size_mb:.2f} MB")
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
    # STEP 2: LLM
    # --------------------------------------------------

    print(
        "\n[2/2] 🧠 Checking + structuring "
        "with GPT-OSS 120B..."
    )

    try:

        llm_start = time.perf_counter()

        structured_data = extract_structured_data(
            ocr_result.raw_text
        )

        llm_time = time.perf_counter() - llm_start

        print(
            f"⏱️ GPT-OSS time: "
            f"{llm_time:.2f} seconds"
        )

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

    # ==================================================
    # RESOURCE MONITORING - END
    # ==================================================

    end_time = time.perf_counter()

    end_ram = process.memory_info().rss / (1024 * 1024)

    ram_change = end_ram - start_ram

    total_time = end_time - start_time

    print("\n" + "=" * 50)
    print("           📊 RESOURCE USAGE")
    print("=" * 50)

    print(f"📦 Image Size : {image_size_mb:.2f} MB")

    print(f"💾 RAM Before : {start_ram:.2f} MB")

    print(f"💾 RAM After  : {end_ram:.2f} MB")

    print(f"📈 RAM Change : {ram_change:+.2f} MB")

    print(f"🔍 OCR Time   : {ocr_time:.2f} seconds")

    print(f"🧠 LLM Time   : {llm_time:.2f} seconds")

    print(f"⏱️ Total Time : {total_time:.2f} seconds")

    print("=" * 50)


# ======================================================
# WINDOWS UTF-8 SUPPORT
# ======================================================

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):

    sys.stdout.reconfigure(
        encoding="utf-8"
    )


# ======================================================
# FIND ALL IMAGES
# ======================================================

def get_all_images():
    """Find all image files in input folders, sorted newest first."""

    search_dirs = [
        "inputs",
        "Inout",
        "input",
        "inout"
    ]

    valid_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".webp"
    }

    found = []

    seen = set()

    for d in search_dirs:

        if os.path.exists(d):

            for root, _, files in os.walk(d):

                for f in files:

                    if os.path.splitext(f)[1].lower() in valid_exts:

                        fp = os.path.normpath(
                            os.path.join(root, f)
                        )

                        if fp not in seen:

                            seen.add(fp)

                            found.append(
                                (
                                    os.path.getmtime(fp),
                                    fp
                                )
                            )

    # Sort newest first

    found.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        fp
        for _, fp in found
    ]


# ======================================================
# PRINT IMAGE LIST
# ======================================================

def print_image_list(images):
    """Print numbered list of available images."""

    print("\n📋 Available Images in Folder:")

    print("-" * 50)

    for idx, img in enumerate(images, 1):

        latest_tag = (
            " (Latest)"
            if idx == 1
            else ""
        )

        print(
            f"  [{idx}] {img}{latest_tag}"
        )

    print("-" * 50)

    print(
        "💡 To run an image by number, "
        "use: python main.py <number>"
    )

    print(
        "   Example: python main.py 2\n"
    )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Extract structured data "
            "from a business card image."
        )
    )

    parser.add_argument(
        "arg",
        nargs="?",
        help=(
            "Image number (e.g. 1, 2), "
            "file path, 'camera', or 'list'"
        )
    )

    args = parser.parse_args()

    user_arg = args.arg

    # --------------------------------------------------
    # CAMERA COMMAND
    # --------------------------------------------------

    if user_arg and user_arg.lower() in [
        "camera",
        "cam"
    ]:

        try:

            from inputs.capture import capture_image

        except ImportError:

            try:

                from camera.capture import capture_image

            except ImportError:

                print(
                    "❌ Could not import capture_image "
                    "from inputs.capture or camera.capture"
                )

                sys.exit(1)

        captured_path = capture_image()

        if captured_path:

            process_business_card(
                captured_path
            )

        sys.exit(0)

    # --------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------

    images = get_all_images()

    if not images:

        print(
            "❌ No images found in "
            "inputs/ or Inout/ folder!"
        )

        print(
            "💡 Place your business card images "
            "inside 'inputs' or 'Inout' folder "
            "or use 'python main.py camera'."
        )

        sys.exit(1)

    # --------------------------------------------------
    # LIST COMMAND
    # --------------------------------------------------

    if user_arg in [
        "list",
        "-l",
        "--list"
    ]:

        print_image_list(images)

        sys.exit(0)

    # --------------------------------------------------
    # SELECT IMAGE
    # --------------------------------------------------

    image_path = None

    # No argument → latest image

    if not user_arg:

        image_path = images[0]

        print(
            f"ℹ️ No image specified. "
            f"Auto-selected latest image [#1]: "
            f"{image_path}"
        )

        print(
            "💡 Run 'python main.py list' "
            "to see all numbered images "
            "or 'python main.py camera' "
            "to capture live."
        )

    # Number argument

    elif user_arg.isdigit():

        num = int(user_arg)

        if 1 <= num <= len(images):

            image_path = images[num - 1]

            print(
                f"ℹ️ Selected image #{num}: "
                f"{image_path}"
            )

        else:

            print(
                f"❌ Invalid image number: {num}. "
                f"Must be between 1 and {len(images)}."
            )

            print_image_list(images)

            sys.exit(1)

    # Direct path / filename

    else:

        if os.path.exists(user_arg):

            image_path = user_arg

        else:

            matches = [
                img
                for img in images
                if user_arg.lower() in img.lower()
            ]

            if len(matches) == 1:

                image_path = matches[0]

                print(
                    f"ℹ️ Matched image: "
                    f"{image_path}"
                )

            elif len(matches) > 1:

                print(
                    f"⚠️ Multiple images match "
                    f"'{user_arg}':"
                )

                for m in matches:

                    print(f"  - {m}")

                print(
                    "\nPlease be more specific "
                    "or use the image number."
                )

                sys.exit(1)

            else:

                print(
                    f"❌ File not found: "
                    f"'{user_arg}'"
                )

                print_image_list(images)

                sys.exit(1)

    # --------------------------------------------------
    # PROCESS BUSINESS CARD
    # --------------------------------------------------

    process_business_card(
        image_path
    )
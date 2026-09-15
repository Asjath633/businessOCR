import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

WINDOW_NAME = "Business Card Scanner"

OUTPUT_PATH = "inputs/camera_capture.jpg"

TARGET_WIDTH = 1200


# ============================================================
# GUIDE SETTINGS
# ============================================================

# Business cards are normally around 1.7 - 1.8 : 1

GUIDE_WIDTH_RATIO = 0.70
GUIDE_HEIGHT_RATIO = 0.40


# ============================================================
# GET GUIDE RECTANGLE
# ============================================================

def get_guide_rectangle(frame):
    """
    Create a business-card-shaped guide rectangle.
    """

    height, width = frame.shape[:2]

    guide_width = int(
        width * GUIDE_WIDTH_RATIO
    )

    guide_height = int(
        guide_width / 1.75
    )

    # Safety check
    max_height = int(
        height * GUIDE_HEIGHT_RATIO
    )

    if guide_height > max_height:
        guide_height = max_height

    x1 = (width - guide_width) // 2
    y1 = (height - guide_height) // 2

    x2 = x1 + guide_width
    y2 = y1 + guide_height

    return x1, y1, x2, y2


# ============================================================
# ORDER FOUR POINTS
# ============================================================

def order_points(points):
    """
    Order points as:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    total = points.sum(
        axis=1
    )

    difference = np.diff(
        points,
        axis=1
    ).reshape(-1)

    ordered[0] = points[
        np.argmin(total)
    ]

    ordered[2] = points[
        np.argmax(total)
    ]

    ordered[1] = points[
        np.argmin(difference)
    ]

    ordered[3] = points[
        np.argmax(difference)
    ]

    return ordered


# ============================================================
# TRY TO FIND CARD INSIDE GUIDE
# ============================================================

def find_card_inside_guide(frame):
    """
    Try to find the business card boundary.

    This is OPTIONAL.

    Failure is completely okay because the guide crop
    will be used as a fallback.
    """

    x1, y1, x2, y2 = get_guide_rectangle(
        frame
    )

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        30,
        120
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    roi_area = (
        roi.shape[0] *
        roi.shape[1]
    )

    best_quad = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Card should occupy reasonable portion of guide
        if area < roi_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        points = approx.reshape(
            4,
            2
        )

        x, y, w, h = cv2.boundingRect(
            points
        )

        if w <= 0 or h <= 0:
            continue

        aspect_ratio = (
            max(w, h) /
            min(w, h)
        )

        if not (
            1.25 <= aspect_ratio <= 2.30
        ):
            continue

        if area > best_area:

            best_area = area
            best_quad = points

    if best_quad is None:
        return None

    # Convert ROI coordinates back to frame coordinates
    best_quad[:, 0] += x1
    best_quad[:, 1] += y1

    return best_quad.astype(
        np.float32
    )


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def perspective_crop(frame, points):
    """
    Straighten the card using its four detected corners.
    """

    points = order_points(
        points
    )

    top_left = points[0]
    top_right = points[1]
    bottom_right = points[2]
    bottom_left = points[3]

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    max_width = int(
        max(
            width_top,
            width_bottom
        )
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    max_height = int(
        max(
            height_left,
            height_right
        )
    )

    if (
        max_width <= 0
        or max_height <= 0
    ):
        return None

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    warped = cv2.warpPerspective(
        frame,
        matrix,
        (
            max_width,
            max_height
        )
    )

    return warped


# ============================================================
# GUIDE CROP
# ============================================================

def crop_guide(frame):
    """
    Reliable fallback.

    Simply crops the region inside the guide.
    """

    x1, y1, x2, y2 = get_guide_rectangle(
        frame
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# PREPARE FOR OCR
# ============================================================

def prepare_for_ocr(card):
    """
    Resize card for EasyOCR.
    """

    if card is None:
        return None

    height, width = card.shape[:2]

    if (
        width <= 0
        or height <= 0
    ):
        return None

    if width < TARGET_WIDTH:

        scale = (
            TARGET_WIDTH /
            width
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        card = cv2.resize(
            card,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_CUBIC
        )

    return card


# ============================================================
# SHARPNESS
# ============================================================

def calculate_sharpness(image):
    """
    Estimate image sharpness.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


# ============================================================
# DRAW GUIDE
# ============================================================

def draw_guide(frame):
    """
    Draw guide and instructions.
    """

    x1, y1, x2, y2 = get_guide_rectangle(
        frame
    )

    # White guide
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "ALIGN BUSINESS CARD HERE",
        (x1, y1 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = CAPTURE",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "ESC = CANCEL",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# CAMERA CAPTURE
# ============================================================

def capture_image(
    output_path=OUTPUT_PATH
):
    """
    Open webcam and capture a business card.

    SPACE:
        Capture

    ESC:
        Cancel

    Card detection is optional.
    Guide crop is always available.
    """

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    # Fallback camera opening
    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )

    # ========================================================
    # CAMERA SETTINGS
    # ========================================================

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print()
    print("=" * 60)
    print("📷 BUSINESS CARD SCANNER")
    print("=" * 60)
    print()
    print("Align the complete business card inside the guide.")
    print()
    print("SPACE = Capture")
    print("ESC   = Cancel")
    print()
    print("Card detection is automatic when possible.")
    print("Guide crop is used if detection fails.")
    print()
    print("=" * 60)

    while True:

        ret, frame = camera.read()

        if not ret:

            camera.release()
            cv2.destroyAllWindows()

            raise RuntimeError(
                "Could not read frame from camera."
            )

        # ----------------------------------------------------
        # Try optional card detection
        # ----------------------------------------------------

        card_points = find_card_inside_guide(
            frame
        )

        display = frame.copy()

        # ----------------------------------------------------
        # Draw guide
        # ----------------------------------------------------

        display = draw_guide(
            display
        )

        # ----------------------------------------------------
        # If card detected, show green outline
        # ----------------------------------------------------

        if card_points is not None:

            points = order_points(
                card_points
            ).astype(np.int32)

            cv2.polylines(
                display,
                [points],
                True,
                (0, 255, 0),
                4
            )

            cv2.putText(
                display,
                "CARD FOUND",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # Show camera
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # SPACE
        # ====================================================

        if key == 32:

            print()
            print("📸 Capturing...")

            card = None

            # ------------------------------------------------
            # OPTION 1
            # Detected card
            # ------------------------------------------------

            if card_points is not None:

                print(
                    "✓ Card boundary detected."
                )

                card = perspective_crop(
                    frame,
                    card_points
                )

            # ------------------------------------------------
            # OPTION 2
            # Guide crop
            # ------------------------------------------------

            if card is None:

                print(
                    "↪ Using guide crop."
                )

                card = crop_guide(
                    frame
                )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if card is None:

                print(
                    "❌ Could not capture card."
                )

                continue

            # ------------------------------------------------
            # Sharpness
            # ------------------------------------------------

            sharpness = calculate_sharpness(
                card
            )

            print(
                f"📊 Sharpness: {sharpness:.1f}"
            )

            if sharpness < 30:

                print(
                    "⚠️ Warning: image may be blurry."
                )

            # ------------------------------------------------
            # Prepare for OCR
            # ------------------------------------------------

            card = prepare_for_ocr(
                card
            )

            if card is None:

                print(
                    "❌ Could not prepare image."
                )

                continue

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            Path(
                output_path
            ).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            success = cv2.imwrite(
                output_path,
                card,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            if not success:

                camera.release()
                cv2.destroyAllWindows()

                raise RuntimeError(
                    "Could not save captured image."
                )

            # ------------------------------------------------
            # Done
            # ------------------------------------------------

            print()
            print(
                f"✅ Card saved: {output_path}"
            )

            print(
                "➡️ Sending card image to EasyOCR..."
            )

            camera.release()
            cv2.destroyAllWindows()

            return output_path

        # ====================================================
        # ESC
        # ====================================================

        elif key == 27:

            camera.release()
            cv2.destroyAllWindows()

            print()
            print(
                "❌ Capture cancelled."
            )

            return None
import sys
import time
import json
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# BACKEND
# ============================================================

from ocr.extractor import extract_text_from_image
from llm.extractor import extract_structured_data


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Business Card AI",
    page_icon="🪪",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application width */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .app-title {
        font-size: 38px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .app-subtitle {
        color: #888;
        font-size: 16px;
    }

    /* Section titles */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #888;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Camera */
    [data-testid="stCameraInput"] {
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stCameraInput"] video {
        max-height: 430px;
        object-fit: contain;
    }

    [data-testid="stCameraInput"] img {
        max-height: 430px;
        object-fit: contain;
    }

    /* Uploaded image */
    [data-testid="stImage"] img {
        max-height: 430px;
        object-fit: contain;
    }

    /* Result cards */
    .result-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }

    .value {
        font-size: 17px;
        font-weight: 500;
    }

    /* Small helper text */
    .helper {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 8px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🪪 Business Card AI</div>
        <div class="app-subtitle">
            Extract business card information using AI
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT METHOD
# ============================================================

camera_tab, upload_tab = st.tabs(
    [
        "📷 Camera",
        "📁 Upload",
    ]
)


image_file = None


# ============================================================
# CAMERA TAB
# ============================================================

with camera_tab:

    st.markdown(
        '<div class="section-title">📷 Scan a Business Card</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Keep the entire card visible and make sure the text is clear.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Center camera
    left, center, right = st.columns(
        [1, 3, 1]
    )

    with center:

        camera_image = st.camera_input(
            "Take a picture",
            label_visibility="collapsed",
        )

        if camera_image is not None:

            image_file = camera_image

            st.markdown(
                '<div class="helper">✓ Photo captured</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# UPLOAD TAB
# ============================================================

with upload_tab:

    st.markdown(
        '<div class="section-title">📁 Upload a Business Card</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload JPG, PNG or WEBP images.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_image is not None:

        image_file = uploaded_image

        st.image(
            uploaded_image,
            width=650,
        )


# ============================================================
# PROCESSING
# ============================================================

if image_file is not None:

    st.write("")

    # ========================================================
    # EXTRACT BUTTON
    # ========================================================

    _, button_col, _ = st.columns(
        [1, 2, 1]
    )

    with button_col:

        extract_button = st.button(
            "✨ Extract Information",
            type="primary",
            use_container_width=True,
        )


    # ========================================================
    # RUN AI PIPELINE
    # ========================================================

    if extract_button:

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        inputs_dir = ROOT_DIR / "inputs"

        inputs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = (
            inputs_dir /
            "streamlit_capture.jpg"
        )

        with open(
            image_path,
            "wb",
        ) as f:

            f.write(
                image_file.getbuffer()
            )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        with st.status(
            "🤖 Processing your business card...",
            expanded=True,
        ) as status:

            # ================================================
            # OCR
            # ================================================

            st.write(
                "🔍 Reading text with PP-OCRv6..."
            )

            ocr_start = time.perf_counter()

            ocr_result = extract_text_from_image(
                str(image_path)
            )

            ocr_time = (
                time.perf_counter()
                - ocr_start
            )

            raw_text = ocr_result.raw_text
            confidence = ocr_result.confidence


            # ================================================
            # GPT-OSS
            # ================================================

            st.write(
                "🧠 Structuring data with GPT-OSS 120B..."
            )

            llm_start = time.perf_counter()

            structured_data = (
                extract_structured_data(
                    raw_text
                )
            )

            llm_time = (
                time.perf_counter()
                - llm_start
            )


            # ================================================
            # COMPLETE
            # ================================================

            status.update(
                label="✅ Extraction completed",
                state="complete",
            )


        # ====================================================
        # BUSINESS CARD CHECK
        # ====================================================

        if not structured_data.is_business_card:

            st.warning(
                "This image doesn't appear to be a business card."
            )

            with st.expander(
                "🔍 View detected OCR text"
            ):

                st.text(raw_text)

            st.stop()


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "✅ Business card detected successfully!"
        )


        st.divider()


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">📋 Extracted Information</div>',
            unsafe_allow_html=True,
        )

        st.write("")


        names = structured_data.name or []
        emails = structured_data.email or []
        phones = structured_data.phone or []


        name_value = (
            ", ".join(names)
            if names
            else "Not found"
        )

        email_value = (
            ", ".join(emails)
            if emails
            else "Not found"
        )

        phone_value = (
            ", ".join(phones)
            if phones
            else "Not found"
        )

        organization = (
            structured_data.organization
            or "Not found"
        )

        designation = (
            structured_data.designation
            or "Not found"
        )

        address = (
            structured_data.address
            or "Not found"
        )


        # ====================================================
        # RESULT GRID
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            with st.container(border=True):

                st.markdown(
                    '<div class="label">👤 Name</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="value">{name_value}</div>',
                    unsafe_allow_html=True,
                )


            with st.container(border=True):

                st.markdown(
                    '<div class="label">🏢 Organization</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="value">{organization}</div>',
                    unsafe_allow_html=True,
                )


            with st.container(border=True):

                st.markdown(
                    '<div class="label">💼 Designation</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="value">{designation}</div>',
                    unsafe_allow_html=True,
                )


        with col2:

            with st.container(border=True):

                st.markdown(
                    '<div class="label">📞 Phone</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="value">{phone_value}</div>',
                    unsafe_allow_html=True,
                )


            with st.container(border=True):

                st.markdown(
                    '<div class="label">📧 Email</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="value">{email_value}</div>',
                    unsafe_allow_html=True,
                )


            with st.container(border=True):

                st.markdown(
                    '<div class="label">📍 Address</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="value">{address}</div>',
                    unsafe_allow_html=True,
                )


        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.write("")

        st.markdown(
            '<div class="section-title">⚡ Performance</div>',
            unsafe_allow_html=True,
        )

        total_time = (
            ocr_time +
            llm_time
        )


        m1, m2, m3, m4 = st.columns(4)


        with m1:
            st.metric(
                "OCR",
                f"{ocr_time:.1f}s",
            )


        with m2:
            st.metric(
                "GPT-OSS",
                f"{llm_time:.1f}s",
            )


        with m3:
            st.metric(
                "Total",
                f"{total_time:.1f}s",
            )


        with m4:
            st.metric(
                "Confidence",
                f"{confidence:.1f}%",
            )


        # ====================================================
        # RAW OCR
        # ====================================================

        st.write("")

        with st.expander(
            "🔍 View Raw OCR"
        ):

            st.text(raw_text)


        # ====================================================
        # JSON
        # ====================================================

        with st.expander(
            "🧾 View JSON"
        ):

            json_data = (
                structured_data.model_dump()
            )

            st.json(json_data)


        # ====================================================
        # DOWNLOAD
        # ====================================================

        json_string = json.dumps(
            structured_data.model_dump(),
            indent=2,
            ensure_ascii=False,
        )


        st.download_button(
            "⬇️ Download JSON",
            data=json_string,
            file_name="business_card.json",
            mime="application/json",
            use_container_width=True,
        )
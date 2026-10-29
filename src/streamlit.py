from pathlib import Path
from io import BytesIO
import os
import tempfile

import streamlit as st
from PIL import Image, UnidentifiedImageError

import predict_sinhala_v3 as text_classifier

from image_ocr_v4_6_hybrid_multilingual import (
    run_multilingual_ocr,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered Fake News Detection",
    page_icon="📰",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

ALLOWED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
}

ALLOWED_PDF_EXTENSIONS = {
    ".pdf",
}

ALLOWED_PDF_MIME_TYPES = {
    "application/pdf",
}

MAX_FILE_SIZE_MB = 10


# ============================================================
# SESSION STATE
# ============================================================

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

if "text_result" not in st.session_state:
    st.session_state.text_result = None


if "image_ocr_result" not in st.session_state:
    st.session_state.image_ocr_result = None

if "image_temp_path" not in st.session_state:
    st.session_state.image_temp_path = None

if "image_upload_version" not in st.session_state:
    st.session_state.image_upload_version = 0


# ============================================================
# CLEAR TEXT
# ============================================================

def clear_text_input():

    st.session_state.news_text = ""

    st.session_state.text_result = None


# ============================================================
# CLEAR IMAGE
# ============================================================

def clear_image_input():

    # --------------------------------------------------------
    # REMOVE OLD TEMPORARY IMAGE
    # --------------------------------------------------------

    old_temp_path = (
        st.session_state.get(
            "image_temp_path"
        )
    )

    if old_temp_path:

        try:

            old_path = Path(
                old_temp_path
            )

            if old_path.exists():

                old_path.unlink()

        except Exception:

            pass

    # --------------------------------------------------------
    # CLEAR OCR RESULT
    # --------------------------------------------------------

    st.session_state.image_ocr_result = None

    st.session_state.image_temp_path = None

    # --------------------------------------------------------
    # CHANGE UPLOADER KEY
    #
    # This clears the Streamlit file uploader.
    # --------------------------------------------------------

    st.session_state.image_upload_version += 1


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .supported-box {
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .real-result {
        padding: 14px;
        border-radius: 10px;
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.50);
        font-size: 20px;
        font-weight: bold;
    }

    .fake-result {
        padding: 14px;
        border-radius: 10px;
        background-color: rgba(220, 53, 69, 0.15);
        border: 1px solid rgba(220, 53, 69, 0.50);
        font-size: 20px;
        font-weight: bold;
    }

    .ocr-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        margin-top: 10px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEXT CLASSIFIER
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def initialize_text_classifier():

    text_classifier.load_english_model()

    return text_classifier


# ============================================================
# FILE SIZE
# ============================================================

def get_file_size_mb(
    uploaded_file
):

    size_bytes = len(
        uploaded_file.getvalue()
    )

    return (
        size_bytes
        / (1024 * 1024)
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image_file(
    uploaded_file
):

    if uploaded_file is None:

        return (
            False,
            "Please upload an image first.",
        )

    extension = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
    )

    mime_type = (
        uploaded_file.type
        or ""
    )

    # --------------------------------------------------------
    # EXTENSION
    # --------------------------------------------------------

    if (
        extension
        not in
        ALLOWED_IMAGE_EXTENSIONS
    ):

        return (
            False,
            "Only PNG, JPG and JPEG image files are allowed.",
        )

    # --------------------------------------------------------
    # MIME TYPE
    # --------------------------------------------------------

    if (
        mime_type
        not in
        ALLOWED_IMAGE_MIME_TYPES
    ):

        return (
            False,
            "Invalid image file type.",
        )

    # --------------------------------------------------------
    # FILE SIZE
    # --------------------------------------------------------

    if (
        get_file_size_mb(
            uploaded_file
        )
        > MAX_FILE_SIZE_MB
    ):

        return (
            False,
            f"Maximum image size is "
            f"{MAX_FILE_SIZE_MB} MB.",
        )

    # --------------------------------------------------------
    # VERIFY IMAGE CONTENT
    # --------------------------------------------------------

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )

        image = Image.open(
            BytesIO(
                file_bytes
            )
        )

        image.verify()

        image_format = (
            image.format
            or ""
        ).upper()

        if (
            image_format
            not in
            {
                "PNG",
                "JPEG",
            }
        ):

            return (
                False,
                "The uploaded file is not a supported image.",
            )

    except (
        UnidentifiedImageError,
        OSError,
    ):

        return (
            False,
            "The uploaded file is not a valid image.",
        )

    return (
        True,
        "Valid image file.",
    )


# ============================================================
# PDF VALIDATION
# ============================================================

def validate_pdf_file(
    uploaded_file
):

    if uploaded_file is None:

        return (
            False,
            "Please upload a PDF first.",
        )

    extension = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
    )

    mime_type = (
        uploaded_file.type
        or ""
    )

    if (
        extension
        not in
        ALLOWED_PDF_EXTENSIONS
    ):

        return (
            False,
            "Only PDF files are allowed.",
        )

    if (
        mime_type
        not in
        ALLOWED_PDF_MIME_TYPES
    ):

        return (
            False,
            "Invalid PDF file type.",
        )

    if (
        get_file_size_mb(
            uploaded_file
        )
        > MAX_FILE_SIZE_MB
    ):

        return (
            False,
            f"Maximum PDF size is "
            f"{MAX_FILE_SIZE_MB} MB.",
        )

    file_bytes = (
        uploaded_file.getvalue()
    )

    if not file_bytes.startswith(
        b"%PDF-"
    ):

        return (
            False,
            "The uploaded file is not a valid PDF.",
        )

    return (
        True,
        "Valid PDF file.",
    )


# ============================================================
# SAVE TEMPORARY FILE
# ============================================================

def save_temporary_file(
    uploaded_file,
    suffix,
):

    temporary_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )
    )

    temporary_file.write(
        uploaded_file.getvalue()
    )

    temporary_file.close()

    return (
        temporary_file.name
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        AI-Powered Multi-Modal Fake News Detection
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
         
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="supported-box">
        <b>Supported Inputs:</b>
        Text, Image (PNG/JPG/JPEG), and PDF only.
        Other file types are not accepted.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

text_tab, image_tab, pdf_tab = (
    st.tabs(
        [
            "📝 Text",
            "🖼️ Image",
            "📄 PDF",
        ]
    )
)


# ============================================================
# TEXT TAB
# ============================================================

with text_tab:

    st.header(
        "Text-Based Fake News Detection"
    )

    st.write(
        "Paste the complete Sinhala or English "
        "news article, claim, Facebook post, "
        "or social media text."
    )

    news_text = st.text_area(
        "News Text",
        height=260,
        placeholder=(
            "Paste Sinhala or English news text here..."
        ),
        key="news_text",
    )

    analyze_column, clear_column = (
        st.columns(
            [
                3,
                1,
            ]
        )
    )

    with analyze_column:

        analyze_text_button = (
            st.button(
                "Analyze Text",
                type="primary",
                use_container_width=True,
                key="analyze_text",
            )
        )

    with clear_column:

        st.button(
            "Clear",
            use_container_width=True,
            key="clear_text",
            on_click=clear_text_input,
        )

    # ========================================================
    # TEXT CLASSIFICATION
    # ========================================================

    if analyze_text_button:

        cleaned_text = (
            news_text.strip()
        )

        if not cleaned_text:

            st.error(
                "Please enter news text before analysis."
            )

            st.session_state.text_result = None

        elif len(
            cleaned_text
        ) < 5:

            st.error(
                "The entered text is too short."
            )

            st.session_state.text_result = None

        else:

            try:

                with st.spinner(
                    "Analyzing news text..."
                ):

                    classifier = (
                        initialize_text_classifier()
                    )

                    (
                        detected_language,
                        prediction,
                        confidence,
                    ) = (
                        classifier.predict_news(
                            cleaned_text
                        )
                    )

                st.session_state.text_result = {

                    "language":
                        detected_language,

                    "prediction":
                        prediction,

                    "confidence":
                        confidence,

                    "submitted_text":
                        cleaned_text,
                }

            except Exception as error:

                st.session_state.text_result = None

                st.error(
                    "Text classification failed."
                )

                st.exception(
                    error
                )

    # ========================================================
    # SHOW TEXT RESULT
    # ========================================================

    text_result = (
        st.session_state.text_result
    )

    if text_result is not None:

        st.markdown(
            """
            <div class="result-title">
                Analysis Result
            </div>
            """,
            unsafe_allow_html=True,
        )

        (
            language_column,
            prediction_column,
            confidence_column,
        ) = st.columns(
            3
        )

        with language_column:

            st.metric(
                "Detected Language",
                text_result[
                    "language"
                ],
            )

        with prediction_column:

            st.metric(
                "Prediction",
                text_result[
                    "prediction"
                ],
            )

        with confidence_column:

            st.metric(
                "Confidence",
                text_result[
                    "confidence"
                ],
            )

        if (
            text_result[
                "prediction"
            ]
            == "REAL"
        ):

            st.success(
                "✅ Prediction: REAL NEWS"
            )

        else:

            st.error(
                "⚠️ Prediction: FAKE NEWS"
            )

        with st.expander(
            "View Input Details"
        ):

            st.info(
                text_result[
                    "submitted_text"
                ]
            )

        st.caption(
            "Confidence represents the model's "
            "classification confidence and should "
            "not be interpreted as absolute proof."
        )


# ============================================================
# IMAGE TAB
# ============================================================

with image_tab:

    st.header(
        "Image-Based OCR Text Extraction"
    )

    st.write(
        "Upload a Sinhala or English news image. "
        "The system will extract the visible news text."
    )

    # --------------------------------------------------------
    # FILE UPLOADER
    # --------------------------------------------------------

    uploaded_image = (
        st.file_uploader(
            "Upload News Image",
            type=[
                "png",
                "jpg",
                "jpeg",
            ],
            accept_multiple_files=False,
            key=(
                "image_uploader_"
                + str(
                    st.session_state.image_upload_version
                )
            ),
        )
    )

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    if uploaded_image is not None:

        (
            valid,
            message,
        ) = validate_image_file(
            uploaded_image
        )

        if valid:

            st.success(
                "Image accepted successfully."
            )

            st.image(
                uploaded_image,
                caption=uploaded_image.name,
                use_container_width=True,
            )

            (
                image_info_1,
                image_info_2,
            ) = st.columns(
                2
            )

            with image_info_1:

                st.write(
                    "**File Name:**"
                )

                st.write(
                    uploaded_image.name
                )

            with image_info_2:

                st.write(
                    "**File Size:**"
                )

                st.write(
                    f"{get_file_size_mb(uploaded_image):.2f} MB"
                )

        else:

            st.error(
                message
            )

    # --------------------------------------------------------
    # IMAGE BUTTONS
    # --------------------------------------------------------

    (
        analyze_image_column,
        clear_image_column,
    ) = st.columns(
        [
            3,
            1,
        ]
    )

    with analyze_image_column:

        analyze_image_button = (
            st.button(
                "Extract Text from Image",
                type="primary",
                use_container_width=True,
                key="analyze_image",
            )
        )

    with clear_image_column:

        st.button(
            "Clear",
            use_container_width=True,
            key="clear_image",
            on_click=clear_image_input,
        )

    # ========================================================
    # RUN OCR
    # ========================================================

    if analyze_image_button:

        (
            valid,
            message,
        ) = validate_image_file(
            uploaded_image
        )

        if not valid:

            st.error(
                message
            )

        else:

            try:

                # --------------------------------------------
                # REMOVE PREVIOUS TEMP IMAGE
                # --------------------------------------------

                old_temp_path = (
                    st.session_state.get(
                        "image_temp_path"
                    )
                )

                if old_temp_path:

                    try:

                        old_path = Path(
                            old_temp_path
                        )

                        if old_path.exists():

                            old_path.unlink()

                    except Exception:

                        pass

                # --------------------------------------------
                # SAVE NEW TEMP IMAGE
                # --------------------------------------------

                extension = (
                    Path(
                        uploaded_image.name
                    )
                    .suffix
                    .lower()
                )

                temporary_image_path = (
                    save_temporary_file(
                        uploaded_image,
                        extension,
                    )
                )

                st.session_state.image_temp_path = (
                    temporary_image_path
                )

                # --------------------------------------------
                # RUN EXISTING V4.6 MULTILINGUAL OCR
                # --------------------------------------------

                with st.spinner(
                    "Extracting text from image..."
                ):

                    ocr_result = (
                        run_multilingual_ocr(
                            temporary_image_path,
                            language_mode="auto",
                        )
                    )

                st.session_state.image_ocr_result = (
                    ocr_result
                )

            except Exception as error:

                st.session_state.image_ocr_result = None

                st.error(
                    "OCR processing failed."
                )

                st.exception(
                    error
                )


    # ========================================================
    # DISPLAY OCR RESULT
    # ========================================================

    image_result = (
        st.session_state.image_ocr_result
    )

    if image_result is not None:

        st.divider()

        st.markdown(
            """
            <div class="result-title">
                OCR Result
            </div>
            """,
            unsafe_allow_html=True,
        )

        extracted_text = str(
            image_result.get(
                "text",
                ""
            )
        ).strip()

        final_language = str(
            image_result.get(
                "final_language",
                "unknown"
            )
        )

        selected_language = str(
            image_result.get(
                "selected_language",
                "unknown"
            )
        )

        ocr_engine = str(
            image_result.get(
                "engine",
                "-"
            )
        )

        extraction_method = str(
            image_result.get(
                "extraction_method",
                "-"
            )
        )

        ocr_results = (
            image_result.get(
                "ocr_results",
                []
            )
            or []
        )

        # ----------------------------------------------------
        # TOP SUMMARY
        # ----------------------------------------------------

        (
            language_col,
            lines_col,
        ) = st.columns(
            2
        )

        with language_col:

            language_display = (
                "Sinhala"
                if final_language == "si"
                else
                "English"
                if final_language == "en"
                else
                final_language
            )

            st.metric(
                "Detected Language",
                language_display,
            )

        with lines_col:

            st.metric(
                "OCR Lines",
                len(
                    ocr_results
                ),
            )

        # ----------------------------------------------------
        # EXTRACTED TEXT
        # ----------------------------------------------------

        st.subheader(
            "Extracted Text"
        )

        if extracted_text:

            st.text_area(
                "OCR Extracted Text",
                value=extracted_text,
                height=220,
                disabled=True,
                key="ocr_extracted_text_display",
            )

            st.success(
                "Text extraction completed successfully."
            )

        else:

            st.warning(
                "No readable text was extracted "
                "from this image."
            )

        # ----------------------------------------------------
        # TECHNICAL OCR DETAILS
        # ----------------------------------------------------

        with st.expander(
            "View OCR Details"
        ):

            st.write(
                "**Selected Language Route:**",
                selected_language,
            )

            st.write(
                "**Final Detected Language:**",
                final_language,
            )

            st.write(
                "**OCR Engine:**"
            )

            st.write(
                ocr_engine
            )

            st.write(
                "**Extraction Method:**"
            )

            st.write(
                extraction_method
            )

            st.write(
                "**OCR Line Count:**",
                len(
                    ocr_results
                ),
            )

        st.caption(
            "This Image tab currently performs OCR "
            "text extraction only. "
            "REAL/FAKE classification is not applied "
            "to image OCR output at this PP1 stage."
        )


# ============================================================
# PDF TAB
# ============================================================

with pdf_tab:

    st.header(
        "PDF-Based News Analysis"
    )

    st.write(
        "Upload a PDF containing a "
        "Sinhala or English news article."
    )

    uploaded_pdf = (
        st.file_uploader(
            "Upload News PDF",
            type=[
                "pdf",
            ],
            accept_multiple_files=False,
            key="pdf_uploader",
        )
    )

    if uploaded_pdf is not None:

        (
            valid,
            message,
        ) = validate_pdf_file(
            uploaded_pdf
        )

        if valid:

            st.success(
                "PDF accepted successfully."
            )

            (
                pdf_col1,
                pdf_col2,
            ) = st.columns(
                2
            )

            with pdf_col1:

                st.write(
                    "**File Name:**",
                    uploaded_pdf.name,
                )

            with pdf_col2:

                st.write(
                    "**File Size:**",
                    f"{get_file_size_mb(uploaded_pdf):.2f} MB",
                )

        else:

            st.error(
                message
            )

    analyze_pdf_button = (
        st.button(
            "Analyze PDF",
            use_container_width=True,
            key="analyze_pdf",
        )
    )

    if analyze_pdf_button:

        (
            valid,
            message,
        ) = validate_pdf_file(
            uploaded_pdf
        )

        if not valid:

            st.error(
                message
            )

        else:

            temporary_pdf_path = (
                save_temporary_file(
                    uploaded_pdf,
                    ".pdf",
                )
            )

            st.success(
                "PDF validated and accepted."
            )

            st.info(
                "PDF text extraction will be "
                "connected in the next step."
            )

            with st.expander(
                "Temporary Processing Information"
            ):

                st.code(
                    temporary_pdf_path
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PP1 Prototype · "
    "Sinhala & English Fake News Detection"
)
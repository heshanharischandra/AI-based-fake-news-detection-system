"""Bilingual Sinhala/English fake news classification app."""

import json
import re
import threading
import unicodedata
from pathlib import Path

import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from language_validation_service import validate_language


# ============================================================
# MODEL PATHS
# ============================================================

ENGLISH_MODEL_PATH = Path(
    r"D:\FakeNewsResearch\models\final_xlmr_model"
)

SINHALA_MODEL_PATH = Path(
    r"D:\FakeNewsResearch\models\sinhala_v3_20261012_182444_410513"
    r"\export\sinhala_xlmr_model"
)


# ============================================================
# DEVICES AND LABELS
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ENGLISH_LABEL_MAP = {
    0: "FAKE",
    1: "REAL",
}

SINHALA_LABEL_MAP = {
    0: "REAL",
    1: "FAKE",
}


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

tokenizer = None
model = None

tokenizer_si = None
model_si = None

max_length_si = None

_sinhala_lock = threading.Lock()


# ============================================================
# LOAD ENGLISH MODEL
# ============================================================

def load_english_model():

    global tokenizer
    global model

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            ENGLISH_MODEL_PATH,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            ENGLISH_MODEL_PATH,
            local_files_only=True,
        )
    )

    model.to(
        device
    )

    model.eval()

    print(
        "English model loaded successfully!"
    )

    print(
        "English device:",
        device,
    )


# ============================================================
# LOAD SINHALA MODEL
# ============================================================

def load_sinhala_model():

    global tokenizer_si
    global model_si
    global max_length_si

    if model_si is not None:
        return

    with _sinhala_lock:

        if model_si is not None:
            return

        settings_path = (
            SINHALA_MODEL_PATH
            / "inference_settings.json"
        )

        settings = json.loads(
            settings_path.read_text(
                encoding="utf-8"
            )
        )

        length = int(
            settings[
                "max_length"
            ]
        )

        if length <= 0:

            raise ValueError(
                "Invalid Sinhala max_length "
                "in inference settings."
            )

        loaded_tokenizer = (
            AutoTokenizer.from_pretrained(
                SINHALA_MODEL_PATH,
                local_files_only=True,
            )
        )

        loaded_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                SINHALA_MODEL_PATH,
                local_files_only=True,
            )
        )

        if (
            loaded_model.config.label2id
            != {
                "REAL": 0,
                "FAKE": 1,
            }
        ):

            raise ValueError(
                "Sinhala model label mapping "
                "must be REAL=0, FAKE=1."
            )

        # Keep Sinhala model on CPU.
        # English model can use the GPU.
        loaded_model.to(
            "cpu"
        )

        loaded_model.eval()

        tokenizer_si = (
            loaded_tokenizer
        )

        max_length_si = (
            length
        )

        model_si = (
            loaded_model
        )

        print(
            "Sinhala model loaded successfully! "
            "Device: CPU"
        )


# ============================================================
# NORMALIZE SINHALA
# ============================================================

def normalize_sinhala(
    text,
):

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    text = re.sub(
        r"[\u200b\ufeff]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# NEWS PREDICTION
# ============================================================

def predict_news(
    text,
):

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not text or not text.strip():

        return (
            "Not detected",
            "NO INPUT",
            "-",
        )

    # --------------------------------------------------------
    # LANGUAGE VALIDATION
    #
    # Only Sinhala and English are allowed.
    # Tamil and all other languages are rejected before
    # classification.
    # --------------------------------------------------------

    language_result = (
        validate_language(
            text
        )
    )

    if not language_result[
        "accepted"
    ]:

        detected_name = (
            language_result.get(
                "language_name"
            )
            or
            "Unsupported Language"
        )

        return (
            detected_name,
            (
                "REJECTED - "
                "This system only supports "
                "English and Sinhala."
            ),
            "-",
        )

    language_code = (
        language_result[
            "language_code"
        ]
    )

    # ========================================================
    # SINHALA
    # ========================================================

    if language_code == "si":

        language = (
            "Sinhala"
        )

        load_sinhala_model()

        inputs = tokenizer_si(
            normalize_sinhala(
                text
            ),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length_si,
        )

        active_model = (
            model_si
        )

        active_device = (
            "cpu"
        )

        active_labels = (
            SINHALA_LABEL_MAP
        )

    # ========================================================
    # ENGLISH
    # ========================================================

    elif language_code == "en":

        language = (
            "English"
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        active_model = (
            model
        )

        active_device = (
            device
        )

        active_labels = (
            ENGLISH_LABEL_MAP
        )

    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    else:

        return (
            "Unsupported Language",
            (
                "REJECTED - "
                "This system only supports "
                "English and Sinhala."
            ),
            "-",
        )

    # --------------------------------------------------------
    # MOVE INPUTS TO ACTIVE DEVICE
    # --------------------------------------------------------

    inputs = {

        key:
            value.to(
                active_device
            )

        for key, value
        in inputs.items()
    }

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = (
            active_model(
                **inputs
            )
        )

    probabilities = (
        torch.softmax(
            outputs.logits,
            dim=1,
        )
    )

    prediction_id = (
        torch.argmax(
            probabilities,
            dim=1,
        )
        .item()
    )

    confidence = (
        probabilities[
            0
        ][
            prediction_id
        ]
        .item()
    )

    prediction = (
        active_labels[
            prediction_id
        ]
    )

    return (
        language,
        prediction,
        f"{confidence * 100:.2f}%",
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

def main():

    load_english_model()

    demo = gr.Interface(

        fn=predict_news,

        inputs=gr.Textbox(
            lines=12,
            label=(
                "Paste Sinhala or English News Text"
            ),
            placeholder=(
                "Paste Sinhala or English "
                "news content here..."
            ),
        ),

        outputs=[

            gr.Textbox(
                label="Detected Language"
            ),

            gr.Textbox(
                label="Prediction / Status"
            ),

            gr.Textbox(
                label="Confidence"
            ),
        ],

        title=(
            "Sinhala & English "
            "Fake News Detection"
        ),

        description=(
            "This system supports only "
            "Sinhala and English news content. "
            "Tamil and all other languages "
            "will be rejected before classification."
        ),
    )

    demo.launch()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
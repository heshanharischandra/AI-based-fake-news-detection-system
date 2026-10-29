"""Separate bilingual app. Original predict.py and English weights stay untouched."""
import json
import re
import threading
import unicodedata
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ENGLISH_MODEL_PATH = Path(r"D:\FakeNewsResearch\models\final_xlmr_model")
SINHALA_MODEL_PATH = Path(
    r"D:\FakeNewsResearch\models\sinhala_v2_20261012_173040_367561"
    r"\export\sinhala_xlmr_model"
)

# English keeps its original device, mapping, and 256-token input handling.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENGLISH_LABEL_MAP = {0: "FAKE", 1: "REAL"}
SINHALA_LABEL_MAP = {0: "REAL", 1: "FAKE"}

tokenizer = None
model = None
tokenizer_si = None
model_si = None
max_length_si = None
_sinhala_lock = threading.Lock()


def load_english_model():
    global tokenizer, model
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(
        ENGLISH_MODEL_PATH, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        ENGLISH_MODEL_PATH, local_files_only=True
    )
    model.to(device)
    model.eval()
    print("English model loaded successfully!")
    print("English device:", device)


def load_sinhala_model():
    global tokenizer_si, model_si, max_length_si
    if model_si is not None:
        return
    with _sinhala_lock:
        if model_si is not None:
            return
        settings = json.loads(
            (SINHALA_MODEL_PATH / "inference_settings.json").read_text(encoding="utf-8")
        )
        length = int(settings["max_length"])
        if length <= 0:
            raise ValueError("Invalid Sinhala max_length in inference settings.")
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            SINHALA_MODEL_PATH, local_files_only=True
        )
        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            SINHALA_MODEL_PATH, local_files_only=True
        )
        if loaded_model.config.label2id != {"REAL": 0, "FAKE": 1}:
            raise ValueError("Sinhala model label mapping must be REAL=0, FAKE=1.")
        # CPU reserves the 4GB GPU for the original English model.
        loaded_model.to("cpu")
        loaded_model.eval()
        tokenizer_si = loaded_tokenizer
        max_length_si = length
        model_si = loaded_model
        print("Sinhala model loaded successfully! Device: CPU")


def detect_language(text):
    if any("\u0D80" <= ch <= "\u0DFF" for ch in text):
        return "Sinhala"
    return "English"


def normalize_sinhala(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_news(text):
    if not text.strip():
        return "No text entered", "-", "-"

    language = detect_language(text)
    if language == "Sinhala":
        load_sinhala_model()
        inputs = tokenizer_si(
            normalize_sinhala(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length_si,
        )
        active_model = model_si
        active_device = "cpu"
        active_labels = SINHALA_LABEL_MAP
    else:
        # Pass original text unchanged, with the original English settings.
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        active_model = model
        active_device = device
        active_labels = ENGLISH_LABEL_MAP

    inputs = {key: value.to(active_device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = active_model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction_id = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction_id].item()
    return language, active_labels[prediction_id], f"{confidence * 100:.2f}%"


def main():
    load_english_model()
    demo = gr.Interface(
        fn=predict_news,
        inputs=gr.Textbox(
            lines=12,
            label="Paste Sinhala or English News Text",
            placeholder="Paste full news article, Facebook post, or social media text here...",
        ),
        outputs=[
            gr.Textbox(label="Detected Language"),
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Confidence"),
        ],
        title="Sinhala & English Fake News Detection",
        description=(
            "Paste the complete news text and click Submit. "
            "Text containing Sinhala characters uses the Sinhala model, "
            "including mixed Sinhala/English text."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    main()
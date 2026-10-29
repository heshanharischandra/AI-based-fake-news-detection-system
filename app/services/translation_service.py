from app.network import configure_tls

configure_tls()

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.services.language_service import normalize_text

MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None


def load_translation_model():
    global tokenizer, model

    if tokenizer is None:
        print("[TRANSLATION] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if model is None:
        print("[TRANSLATION] Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        ).to(device)
        model.eval()


def translate_to_english(text, language):
    text = normalize_text(text)

    if not text:
        return ""

    if language == "en":
        return text

    if language not in ("si", "si+en"):
        return ""

    load_translation_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False
    ).to(device)
    if inputs["input_ids"].shape[-1] > 256:
        raise ValueError("Headline exceeds translation token limit; review or split it instead of truncating the claim")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=4
        )

    translated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return normalize_text(translated_text)

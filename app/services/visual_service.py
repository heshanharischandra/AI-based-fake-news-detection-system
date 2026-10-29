from app.network import configure_tls

configure_tls()

import torch
import re
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration


MODEL_NAME = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def _location_free_caption(caption):
    """Keep the scene description but mark known model-guessed locations unverified."""
    caption = " ".join(str(caption or "").split()).strip()
    if not caption:
        return ""

    # Captioning models are not geolocation models. Preserve the useful visual
    # description instead of discarding it, but never present a guessed place
    # name as image evidence.
    known_places = r"\b(?:hyderabad|colombo|mumbai|delhi|london|paris|new york|beijing|tokyo|sydney)\b"
    caption = re.sub(known_places, "[unverified place]", caption, flags=re.IGNORECASE)
    return caption


def load_visual_model():
    global processor, model

    if processor is None:
        print("[BLIP] Loading image processor...")
        processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    if model is None:
        print("[BLIP] Loading pretrained model...")
        model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME
        )
        model = model.to(device)
        model.eval()

    print("[BLIP] Model ready.")


def describe_image(image_path):
    load_visual_model()

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    # The fixed "a photo showing" prefix biased caption generation. Caption the
    # image directly and retain beam alternatives to expose uncertainty.
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=5,
            num_return_sequences=3,
            no_repeat_ngram_size=3,
            repetition_penalty=1.12,
            length_penalty=1.05,
            do_sample=False,
        )

    captions = []
    for sequence in output:
        candidate = _location_free_caption(
            processor.decode(sequence, skip_special_tokens=True)
        )
        if candidate and candidate.casefold() not in {item.casefold() for item in captions}:
            captions.append(candidate)
    caption = captions[0] if captions else ""

    return {
        "caption": caption,
        "caption_candidates": captions,
        "caption_count": len(captions),
        "caption_is_verified": False,
        "location_status": "unverified_if_named",
        "model": MODEL_NAME,
        "device": device,
        "status": "success" if caption else "no_caption_generated",
    }

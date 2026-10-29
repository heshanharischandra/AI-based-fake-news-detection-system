
from app.network import configure_tls

configure_tls()

import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor
from app.services.news_context_classifier import MODEL_PATH as CONTEXT_MODEL_PATH, predict_context


MODEL_NAME = "openai/clip-vit-base-patch32"

TEXT_TEMPLATES = (
    "a news photograph showing {claim}",
    "a photograph related to {claim}",
    "an image illustrating {claim}",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def load_clip_model():
    global processor, model

    if processor is None:
        print("[CLIP] Loading image-text processor...")
        processor = CLIPProcessor.from_pretrained(
            MODEL_NAME
        )

    if model is None:
        print("[CLIP] Loading pretrained model...")
        model = CLIPModel.from_pretrained(
            MODEL_NAME
        )

        model = model.to(device)
        model.eval()

    print("[CLIP] Model ready.")


def compare_image_text(image_path, headline):
    if not headline or not headline.strip():
        return {
            "similarity_score": None,
            "status": "no_valid_headline",
            "model": MODEL_NAME
        }

    try:
        load_clip_model()
    except Exception as error:
        return {
            "headline": headline.strip(),
            "similarity_score": None,
            "status": "model_unavailable",
            "model": MODEL_NAME,
            "error": str(error),
            "note": "CLIP could not be loaded; install/cache the model and fix Hugging Face SSL connectivity.",
        }

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    claim = headline.strip()
    prompts = [template.format(claim=claim) for template in TEXT_TEMPLATES]
    use_trained_context = CONTEXT_MODEL_PATH.exists()
    if use_trained_context:
        # Official training embeddings use raw captions, not the demo's
        # prompt templates. Keep inference aligned with that feature recipe.
        prompts.append(claim)
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        scores = (image_features @ text_features[:len(TEXT_TEMPLATES)].T).squeeze(0)

    trained_context = {"status": "context_model_not_trained"}
    if use_trained_context:
        try:
            trained_context = predict_context(
                image_features[0].detach().cpu().numpy(),
                text_features[-1].detach().cpu().numpy(),
            )
        except (ValueError, OSError, KeyError, TypeError) as error:
            trained_context = {"status": "context_model_unavailable", "error": str(error)}

    prompt_scores = [round(float(score), 4) for score in scores]
    similarity = float(scores.mean().item())
    best_score = float(scores.max().item())

    return {
        "headline": claim,
        "trained_context": trained_context,
        "similarity_score": round(similarity, 4),
        "best_template_score": round(best_score, 4),
        "template_scores": dict(zip(TEXT_TEMPLATES, prompt_scores)),
        "status": "similarity_calculated",
        "model": MODEL_NAME,
        "device": device,
        "note": "Aggregated CLIP cosine similarity; not accuracy or a truth probability.",
        "calibration_status": "uncalibrated",
    }

from app.network import configure_tls

configure_tls()

import re

import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

MODEL_NAME = "dima806/ai_vs_real_image_detection"
AI_THRESHOLD = 0.85
REAL_THRESHOLD = 0.85
MAX_AUGMENTATION_DISAGREEMENT = 0.10
AI_LABEL_TOKENS = ("FAKE", "AI", "GENERATED", "SYNTHETIC")
REAL_LABEL_TOKENS = ("REAL", "HUMAN", "AUTHENTIC")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def load_authenticity_model():
    global processor, model

    if processor is None:
        print("[AUTHENTICITY] Loading image processor...")
        processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )

    if model is None:
        print("[AUTHENTICITY] Loading pretrained model...")
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME
        ).to(device)
        model.eval()

    print("[AUTHENTICITY] Model ready.")


def _classify_label(label):
    # Substring matching misreads labels such as PAINTING as containing AI.
    tokens = set(re.findall(r"[A-Z]+", str(label).upper()))
    ai_label = bool(tokens.intersection(AI_LABEL_TOKENS))
    real_label = bool(tokens.intersection(REAL_LABEL_TOKENS))
    if ai_label == real_label or tokens.intersection({"NOT", "NON"}):
        return "unknown"
    if ai_label:
        return "ai_generated"
    if real_label:
        return "real"
    return "unknown"


def _label_for_index(labels, index):
    return labels.get(index, labels.get(str(index), f"LABEL_{index}"))


def _prepare_image(image_path):
    """Use displayed orientation and visible pixels without modifying the file."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB")


def _check_ai_generation(image_path, *, input_contains_text_overlay=False):
    # Reject unreadable inputs before loading the model.
    image = _prepare_image(image_path)
    try:
        load_authenticity_model()
    except Exception as error:
        return {
            "real_score": None,
            "ai_generated_score": None,
            "status": "model_unavailable",
            "predicted_class": "unavailable",
            "model_score": None,
            "verification_status": "not_verified",
            "model": MODEL_NAME,
            "error": str(error),
            "note": "The authenticity model could not be loaded. No authenticity decision was made.",
        }

    # These are sensitivity checks on the SAME model, not independent models
    # or demonstrated accuracy improvements. Retain the original image view.
    variants = [
        ("original", image),
        ("mirrored", ImageOps.mirror(image)),
        ("contrast_adjusted", ImageEnhance.Contrast(image).enhance(1.05)),
    ]
    probability_batches = []
    if getattr(model.config, "problem_type", None) == "multi_label_classification":
        raise ValueError("Expected a mutually exclusive AI-versus-real classifier.")
    with torch.inference_mode():
        for _, variant in variants:
            inputs = processor(images=variant, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.ndim != 2 or tuple(logits.shape) != (1, 2):
                raise ValueError("Expected one binary-class prediction per image view.")
            if not torch.isfinite(logits).all().item():
                raise ValueError("The model returned non-finite logits.")
            probability_batches.append(torch.softmax(logits.float(), dim=-1)[0])
    probabilities = torch.stack(probability_batches)
    if not torch.isfinite(probabilities).all().item():
        raise ValueError("The model returned non-finite probabilities.")
    mean_probabilities = probabilities.mean(dim=0)

    labels = getattr(model.config, "id2label", {})
    class_labels = [_label_for_index(labels, index) for index in range(len(mean_probabilities))]
    class_kinds = [_classify_label(label) for label in class_labels]
    if "ai_generated" not in class_kinds or "real" not in class_kinds:
        raise ValueError(f"Unexpected model labels: {class_labels}")

    ai_indices = [index for index, kind in enumerate(class_kinds) if kind == "ai_generated"]
    real_indices = [index for index, kind in enumerate(class_kinds) if kind == "real"]
    ai_score = sum(float(mean_probabilities[index].item()) for index in ai_indices)
    real_score = sum(float(mean_probabilities[index].item()) for index in real_indices)

    ai_variation = float(probabilities[:, ai_indices].sum(dim=1).std(unbiased=False).item())
    ai_view_scores = [
        sum(float(probabilities[view_index][index].item()) for index in ai_indices)
        for view_index in range(len(variants))
    ]
    real_view_scores = probabilities[:, real_indices].sum(dim=1).tolist()
    all_ai_high = all(score >= AI_THRESHOLD for score in ai_view_scores)
    all_real_high = all(score >= REAL_THRESHOLD for score in real_view_scores)
    top_index = int(mean_probabilities.argmax().item())
    predicted_class = _classify_label(class_labels[top_index])
    model_score = float(mean_probabilities[top_index].item())

    review_reasons = []
    if input_contains_text_overlay:
        review_reasons.append("Text overlays may influence the classifier; review the photographic region separately.")
    if ai_variation > MAX_AUGMENTATION_DISAGREEMENT:
        review_reasons.append("The classifier is sensitive to small changes in the input image.")
    if not (all_ai_high or all_real_high):
        review_reasons.append("The image views do not consistently meet the same class threshold.")

    if not review_reasons and all_ai_high:
        status = "possible_ai_generated_high_score"
    elif not review_reasons and all_real_high:
        status = "likely_real_high_score"
    else:
        status = "needs_manual_review"

    return {
        "real_score": round(real_score, 4),
        "ai_generated_score": round(ai_score, 4),
        "status": status,
        "predicted_class": predicted_class,
        "predicted_label": str(class_labels[top_index]),
        "model_score": round(model_score, 4),
        "verification_status": "not_verified",
        "requires_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "input_contains_text_overlay": bool(input_contains_text_overlay),
        "decision_threshold": {
            "ai_generated": AI_THRESHOLD,
            "real": REAL_THRESHOLD,
        },
        "score_margin": round(abs(ai_score - real_score), 4),
        "augmentation_disagreement": round(ai_variation, 4),
        "augmentation_count": len(variants),
        "augmentation_scores": [
            {"name": name, "ai_score": round(ai_view_scores[variant_index], 4)}
            for variant_index, (name, _) in enumerate(variants)
        ],
        "label_scores": {
            str(label): round(float(mean_probabilities[index].item()), 4)
            for index, label in enumerate(class_labels)
        },
        "calibration_status": "uncalibrated",
        "model": MODEL_NAME,
        "device": device,
        "note": "The predicted class and score are uncalibrated model outputs, not verified authenticity or accuracy.",
    }


def check_ai_generation(image_path, *, input_contains_text_overlay=False):
    """Return a cautious AI-generation indicator; never treat it as authenticity proof."""
    try:
        result = _check_ai_generation(image_path, input_contains_text_overlay=input_contains_text_overlay)
        scores = (result.get("real_score"), result.get("ai_generated_score"))
        if any(score is not None and not torch.isfinite(torch.tensor(score)) for score in scores):
            raise ValueError("The model returned a non-finite probability score.")
        if all(score is not None for score in scores) and not (
            0.98 <= sum(scores) <= 1.02
        ):
            raise ValueError("The model returned invalid class probabilities.")
        return result
    except Exception as error:
        return {
            "real_score": None,
            "ai_generated_score": None,
            "status": "analysis_unavailable",
            "predicted_class": "unavailable",
            "model_score": None,
            "verification_status": "not_verified",
            "calibration_status": "uncalibrated",
            "model": MODEL_NAME,
            "device": device,
            "error": str(error),
            "note": "No AI-generation indicator was produced. Image authenticity, manipulation, and news-context correctness remain unverified.",
        }




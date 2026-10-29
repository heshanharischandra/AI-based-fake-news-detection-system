
import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "news_context.json"
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "news_context_training.json"


def _benchmark_summary():
    """Expose the saved held-out dataset metric without implying per-image certainty."""
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        test = report["test"]
        if report.get("dataset") != "NewsCLIPpings merged_balanced":
            return None
        accuracy = float(test["accuracy"])
        samples = int(test["samples"])
        if not np.isfinite(accuracy) or samples <= 0:
            return None
        return {
            "dataset": report["dataset"],
            "test_samples": samples,
            "test_accuracy_percent": round(accuracy * 100, 2),
            "test_balanced_accuracy_percent": round(float(test["balanced_accuracy"]) * 100, 2),
            "scope": "constructed original-vs-mismatched English image/caption pairs; not real-news truth or per-image accuracy",
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pair_features(images, texts):
    images = np.asarray(images, dtype=np.float32)
    texts = np.asarray(texts, dtype=np.float32)
    if images.ndim == 1:
        images = images[None, :]
    if texts.ndim == 1:
        texts = texts[None, :]
    if images.shape != texts.shape or images.ndim != 2 or images.shape[1] != 512:
        raise ValueError("Expected matching N x 512 CLIP features")
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError("Nonfinite CLIP features")
    image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    text_norm = np.linalg.norm(texts, axis=1, keepdims=True)
    if np.any(image_norm == 0) or np.any(text_norm == 0):
        raise ValueError("Zero CLIP feature vector")
    images = images / image_norm
    texts = texts / text_norm
    product = images * texts
    return np.concatenate([product * np.sqrt(512), np.abs(images - texts), product.sum(axis=1, keepdims=True)], axis=1).astype(np.float32)


def predict_context(image_embedding, caption_embedding, model_path=MODEL_PATH):
    if not model_path.exists():
        return {"status": "context_model_not_trained"}
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    if (artifact.get("schema_version") != 1
            or artifact.get("encoder") != "openai/clip-vit-base-patch32"
            or artifact.get("feature_version") != "normalized_product_absdiff_cosine_v1"
            or artifact.get("text_mode") != "raw_caption_no_prompt"):
        raise ValueError("Incompatible context-model artifact")
    weights = np.asarray(artifact["coef"], dtype=np.float64)
    threshold = float(artifact["threshold"])
    if weights.shape != (1025,) or not np.isfinite(weights).all() or not 0 < threshold < 1:
        raise ValueError("Invalid context-model weights or threshold")
    features = pair_features(image_embedding, caption_embedding)
    logit = float((features @ weights)[0]) + float(artifact["intercept"])
    if not np.isfinite(logit):
        raise ValueError("Nonfinite model logit")
    score = float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))
    result = {"status": "dataset_model_prediction", "mismatch_score": round(score, 4),
            "prediction": "possible_out_of_context" if score >= threshold else "image_caption_pair_appears_aligned",
            "threshold": threshold, "dataset": artifact["dataset"],
            "calibration_status": artifact["calibration_status"], "verified_evidence": False,
            "note": "NewsCLIPpings model output, not factual verification or measured per-image accuracy."}
    benchmark = _benchmark_summary()
    if benchmark:
        result["benchmark"] = benchmark
    return result

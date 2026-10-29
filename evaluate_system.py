"""Evaluate the deployed multi-modal news-image verification pipeline.

The evaluator expects ground_truth.json in the selected image directory
(original uploads when available, otherwise test_images). The UTF-8 file may
be a list of records or an object with a samples list. Images are relative to
that directory. Use --check-labels to check annotations without loading models.

Minimal record (requires --allow-unlabelled to run predictions only)::

    {"id": "si-001", "image": "si-001.png"}

Record with labels for every requested metric::

    {
      "id": "si-001",
      "image": "sinhala/si-001.png",
      "language": "si",
      "ocr_text": "...exact visible Sinhala headline...",
      "comparison_text": "English text describing the image",
      "clip_match_label": true,
      "reuse_label": false,
      "reuse_expected_reference": null,
      "authenticity_label": "real"
    }

``language`` provides the requested text-classification target: macro F1 of
the OCR language classifier across ``si``, ``en``, ``si+en`` and ``unknown``.
The current application has no independent fake/real *text* classifier, so
the report intentionally does not mislabel a model output as one.

The evaluator does not train, tune, or calibrate production models.  It runs
the deployed services as-is and measures their output against held-out labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import unicodedata
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUPPORTED_LANGUAGES = {"si", "en", "si+en", "unknown"}
AUTHENTICITY_LABELS = {"ai_generated", "real"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DERIVED_UPLOAD_FOLDERS = {"processed", "ui"}
METRIC_LABEL_FIELDS = {
    "text_classification": "language",
    "ocr_character_error_rate": "ocr_text",
    "clip_similarity_threshold": "clip_match_label",
    "image_reuse_retrieval": "reuse_label",
    "ai_authenticity_detection": "authenticity_label",
}


def label_readiness(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep annotation availability separate from model prediction availability."""
    metrics = {}
    for metric, field in METRIC_LABEL_FIELDS.items():
        missing = []
        for sample in samples:
            value = sample.get(field)
            labelled = value is not None
            if field == "ocr_text":
                labelled = isinstance(value, str) and bool(normalize_for_cer(value))
            if not labelled:
                missing.append(sample.get("id", sample.get("image", "unknown")))
        metrics[metric] = {
            "required_field": field,
            "labelled_samples": len(samples) - len(missing),
            "missing_label_sample_ids": missing,
        }
    any_labels = any(item["labelled_samples"] for item in metrics.values())
    complete = bool(samples) and all(not item["missing_label_sample_ids"] for item in metrics.values())
    return {
        "status": "ready" if complete else "partially_labelled" if any_labels else "labels_required",
        "samples": len(samples),
        "metrics": metrics,
    }


def normalize_for_cer(value: Any) -> str:
    """Normalize Unicode and presentation whitespace without transliteration."""
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def levenshtein_distance(reference: str, prediction: str) -> int:
    """Return character-level Levenshtein edit distance using O(min(n, m)) memory."""
    if reference == prediction:
        return 0
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for index, reference_char in enumerate(reference, start=1):
        current = [index]
        for column, prediction_char in enumerate(prediction, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (reference_char != prediction_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def ocr_cer(reference: str, prediction: str) -> tuple[int, int, float | None]:
    """Return edit distance, reference characters, and character error rate."""
    reference = normalize_for_cer(reference)
    prediction = normalize_for_cer(prediction)
    if not reference:
        return 0, 0, None
    distance = levenshtein_distance(reference, prediction)
    return distance, len(reference), distance / len(reference)


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    """Compute binary classification metrics with positive=True."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(actual and predicted for actual, predicted in zip(y_true, y_pred))
    fp = sum(not actual and predicted for actual, predicted in zip(y_true, y_pred))
    tn = sum(not actual and not predicted for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual and not predicted for actual, predicted in zip(y_true, y_pred))
    total = len(y_true)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "evaluated_samples": total,
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
    }


def macro_f1(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Compute deterministic per-class and macro F1 without sklearn at runtime."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for label in labels:
        actual = [value == label for value in y_true]
        predicted = [value == label for value in y_pred]
        metrics = binary_metrics(actual, predicted)
        per_class[label] = metrics
        f1_values.append(metrics["f1"] or 0.0)
    return {
        "evaluated_samples": len(y_true),
        "accuracy": sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true) if y_true else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "labels": labels,
        "per_class": per_class,
    }


def compact_result(result: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Retain stable, useful output fields while keeping reports manageable."""
    return {field: result.get(field) for field in fields if field in result}


def stage_result(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one pipeline stage without losing the rest of the evaluation."""
    started = time.perf_counter()
    try:
        value = operation()
        if not isinstance(value, dict):
            raise TypeError(f"{name} returned {type(value).__name__}, expected a dictionary")
        value = dict(value)
    except Exception as error:  # Pipeline/model errors belong in the report, not a silent skip.
        value = {"status": "stage_error", "error": f"{type(error).__name__}: {error}"}
    value["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return value


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def discover_source_images(test_dir: Path) -> list[Path]:
    """Find user-supplied images while excluding this application's derived output."""
    resolved_root = test_dir.resolve()
    if not resolved_root.is_dir():
        return []
    images: list[Path] = []
    for image in resolved_root.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        relative = image.relative_to(resolved_root)
        if set(relative.parts[:-1]).intersection(DERIVED_UPLOAD_FOLDERS):
            continue
        images.append(image)
    return sorted(images)


def default_test_directory() -> Path:
    """Prefer original upload images when available; otherwise use test_images."""
    uploads = ROOT / "uploads"
    return uploads if discover_source_images(uploads) else ROOT / "test_images"


def create_ground_truth_template(path: Path, test_dir: Path) -> int:
    """Create a non-destructive manifest skeleton from the available images.

    Labels are deliberately omitted.  They must be created from independent,
    trusted annotation rather than copied from the system being evaluated.
    """
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing ground truth: {path}")
    resolved_root = test_dir.resolve()
    images = discover_source_images(resolved_root)
    samples = [
        {
            "id": image.relative_to(resolved_root).with_suffix("").as_posix().replace("/", "-"),
            "image": image.relative_to(resolved_root).as_posix(),
        }
        for image in images
    ]
    template = {
        "schema_version": "ground-truth/v1",
        "annotation_instructions": (
            "Add independent labels described in docs/evaluation_system.md. "
            "Do not use the evaluated model's output as ground truth."
        ),
        "samples": samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(samples)


def load_ground_truth(path: Path, test_dir: Path) -> list[dict[str, Any]]:
    """Load and validate labels before any expensive model is loaded."""
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth JSON was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Ground-truth JSON is invalid: {error}") from error
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError("Ground truth must be a JSON list or an object containing a 'samples' list.")
    if not samples:
        raise ValueError(
            "Ground truth has no samples. Add test images to test_images and complete their independent labels "
            "in ground_truth.json; see docs/evaluation_system.md."
        )

    resolved_root = test_dir.resolve()
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, value in enumerate(samples, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")
        record = dict(value)
        image = record.get("image")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"Sample {index} requires a non-empty relative 'image' path.")
        image_path = Path(image)
        if image_path.is_absolute() or not _is_relative_to((resolved_root / image_path).resolve(), resolved_root):
            raise ValueError(f"Sample {index} image must stay inside test_images: {image!r}")
        sample_id = str(record.get("id") or image)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id!r}")
        seen_ids.add(sample_id)
        record["id"] = sample_id
        record["image"] = image_path.as_posix()
        record["_image_path"] = resolved_root / image_path

        language = record.get("language")
        if language is not None and language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Sample {sample_id!r} has unsupported language {language!r}.")
        for text_key in ("ocr_text", "comparison_text"):
            if text_key in record and not isinstance(record[text_key], str):
                raise ValueError(f"Sample {sample_id!r} field {text_key!r} must be a string.")
        for boolean_key in ("clip_match_label", "reuse_label"):
            if boolean_key in record and not isinstance(record[boolean_key], bool):
                raise ValueError(f"Sample {sample_id!r} field {boolean_key!r} must be true or false.")
        authenticity = record.get("authenticity_label")
        if authenticity is not None and authenticity not in AUTHENTICITY_LABELS:
            raise ValueError(
                f"Sample {sample_id!r} authenticity_label must be one of {sorted(AUTHENTICITY_LABELS)}."
            )
        expected_reference = record.get("reuse_expected_reference")
        if expected_reference is not None and not isinstance(expected_reference, str):
            raise ValueError(f"Sample {sample_id!r} reuse_expected_reference must be a string or null.")
        validated.append(record)
    return validated


def run_pipeline(sample: dict[str, Any]) -> dict[str, Any]:
    """Run the application services in their production evaluation order."""
    # Imports are intentionally lazy: --help and label validation do not trigger model imports.
    from app.services import authenticity_service, claim_service, ocr_service, reuse_service, similarity_service

    image_path = sample["_image_path"]
    if not image_path.is_file():
        return {
            "id": sample["id"],
            "image": sample["image"],
            "ground_truth": {key: sample[key] for key in METRIC_LABEL_FIELDS.values() if key in sample},
            "pipeline_status": "image_missing",
            "error": f"Image not found: {image_path}",
        }

    ocr = stage_result("ocr", lambda: ocr_service.extract_text(image_path))
    claim = stage_result("translation", lambda: claim_service.prepare_news_claim(ocr))
    comparison_text = sample.get("comparison_text") or claim.get("english_claim") or ""
    clip = stage_result("clip", lambda: similarity_service.compare_image_text(image_path, comparison_text))
    authenticity = stage_result(
        "authenticity",
        lambda: authenticity_service.check_ai_generation(
            image_path,
            input_contains_text_overlay=bool(ocr.get("extracted_text")),
        ),
    )
    reuse = stage_result("reuse", lambda: reuse_service.check_image_reuse(image_path))

    return {
        "id": sample["id"],
        "image": sample["image"],
        "pipeline_status": "completed",
        "ground_truth": {
            key: sample.get(key)
            for key in (
                "language", "ocr_text", "comparison_text", "clip_match_label",
                "reuse_label", "reuse_expected_reference", "authenticity_label",
                "annotation_source", "annotation_review_status",
            )
            if key in sample
        },
        "ocr": compact_result(
            ocr,
            ("status", "error", "extracted_text", "language", "confidence", "headline_status", "requires_review", "ocr_engine", "ocr_languages", "elapsed_seconds"),
        ),
        "translation": compact_result(
            claim,
            ("original_claim", "english_claim", "language", "claim_source", "headline_status", "translation_status", "translation_verified", "translation_error", "status", "error", "elapsed_seconds"),
        ),
        "clip": compact_result(
            clip,
            ("status", "similarity_score", "best_template_score", "trained_context", "model", "device", "error", "elapsed_seconds"),
        ) | {"comparison_text_source": "ground_truth" if sample.get("comparison_text") else "pipeline_translation"},
        "authenticity": compact_result(
            authenticity,
            ("status", "predicted_class", "real_score", "ai_generated_score", "model_score", "requires_review", "calibration_status", "model", "device", "error", "elapsed_seconds"),
        ),
        "reuse": compact_result(
            reuse,
            ("status", "possible_reuse", "matched_reference", "nearest_reference_candidate", "match_strength", "reference_count", "failed_references", "error", "elapsed_seconds"),
        ),
    }


def build_summary(records: list[dict[str, Any]], clip_threshold: float) -> dict[str, Any]:
    """Aggregate metrics only from samples with both an evaluable label and prediction."""
    language_true: list[str] = []
    language_pred: list[str] = []
    cer_distances = 0
    cer_reference_characters = 0
    cer_per_sample: list[float] = []
    clip_true: list[bool] = []
    clip_pred: list[bool] = []
    authenticity_true: list[bool] = []
    authenticity_pred: list[bool] = []
    reuse_true: list[bool] = []
    reuse_pred: list[bool] = []
    reuse_reference_total = 0
    reuse_reference_correct = 0
    stage_statuses: dict[str, Counter[str]] = {
        stage: Counter() for stage in ("ocr", "translation", "clip", "authenticity", "reuse")
    }

    for record in records:
        if record.get("pipeline_status") != "completed":
            continue
        truth = record["ground_truth"]
        for stage, counter in stage_statuses.items():
            output = record[stage]
            status = output.get("status", "unknown")
            if stage == "translation" and status != "stage_error":
                status = output.get("translation_status", status)
            counter[str(status)] += 1

        actual_language = truth.get("language")
        predicted_language = record["ocr"].get("language")
        if actual_language in SUPPORTED_LANGUAGES and predicted_language in SUPPORTED_LANGUAGES:
            language_true.append(actual_language)
            language_pred.append(predicted_language)

        if "ocr_text" in truth:
            distance, length, sample_cer = ocr_cer(truth["ocr_text"], record["ocr"].get("extracted_text", ""))
            if sample_cer is not None:
                cer_distances += distance
                cer_reference_characters += length
                cer_per_sample.append(sample_cer)

        score = record["clip"].get("similarity_score")
        if ("clip_match_label" in truth and type(score) in (int, float)
                and math.isfinite(score) and -1 <= score <= 1):
            clip_true.append(truth["clip_match_label"])
            clip_pred.append(float(score) >= clip_threshold)

        actual_authenticity = truth.get("authenticity_label")
        predicted_authenticity = record["authenticity"].get("predicted_class")
        if actual_authenticity in AUTHENTICITY_LABELS and predicted_authenticity in AUTHENTICITY_LABELS:
            authenticity_true.append(actual_authenticity == "ai_generated")
            authenticity_pred.append(predicted_authenticity == "ai_generated")

        actual_reuse = truth.get("reuse_label")
        predicted_reuse = record["reuse"].get("possible_reuse")
        if isinstance(actual_reuse, bool) and isinstance(predicted_reuse, bool):
            reuse_true.append(actual_reuse)
            reuse_pred.append(predicted_reuse)
        expected_reference = truth.get("reuse_expected_reference")
        if actual_reuse is True and expected_reference:
            reuse_reference_total += 1
            if record["reuse"].get("matched_reference") == expected_reference:
                reuse_reference_correct += 1

    summary = {
        "samples_processed": len(records),
        "text_classification": {
            "task": "OCR language classification (Sinhala / English / mixed)",
            **macro_f1(language_true, language_pred),
            "label_coverage": len(language_true) / len(records) if records else 0.0,
        },
        "ocr_character_error_rate": {
            "normalization": "Unicode NFC with whitespace collapsed; Sinhala characters remain Unicode Sinhala.",
            "evaluated_samples": len(cer_per_sample),
            "reference_characters": cer_reference_characters,
            "total_edit_distance": cer_distances,
            "corpus_cer": cer_distances / cer_reference_characters if cer_reference_characters else None,
            "mean_sample_cer": sum(cer_per_sample) / len(cer_per_sample) if cer_per_sample else None,
            "label_coverage": len(cer_per_sample) / len(records) if records else 0.0,
        },
        "clip_similarity_threshold": {
            "threshold": clip_threshold,
            "positive_class": "image and comparison text are aligned",
            **binary_metrics(clip_true, clip_pred),
            "label_coverage": len(clip_true) / len(records) if records else 0.0,
            "calibration_note": "The supplied threshold is not calibrated by this script; select it on separate validation data.",
        },
        "image_reuse_retrieval": {
            "positive_class": "image matches an item in the configured local reference collection",
            **binary_metrics(reuse_true, reuse_pred),
            "label_coverage": len(reuse_true) / len(records) if records else 0.0,
            "top_1_reference_accuracy_on_positive_samples": (
                reuse_reference_correct / reuse_reference_total if reuse_reference_total else None
            ),
            "positive_samples_with_expected_reference": reuse_reference_total,
        },
        "ai_authenticity_detection": {
            "positive_class": "ai_generated",
            **binary_metrics(authenticity_true, authenticity_pred),
            "label_coverage": len(authenticity_true) / len(records) if records else 0.0,
            "note": "Included because the pipeline runs this detector. It is only meaningful when authenticity labels come from trusted provenance.",
        },
        "stage_status_counts": {stage: dict(counter) for stage, counter in stage_statuses.items()},
    }
    readiness = label_readiness([
        dict(record.get("ground_truth", {}), id=record.get("id", str(index)))
        for index, record in enumerate(records)
    ])
    for metric, labels in readiness["metrics"].items():
        result = summary[metric]
        labelled = labels["labelled_samples"]
        evaluated = result["evaluated_samples"]
        result["labelled_samples"] = labelled
        result["missing_label_samples"] = len(records) - labelled
        result["label_coverage"] = labelled / len(records) if records else 0.0
        result["prediction_coverage_on_labelled_samples"] = evaluated / labelled if labelled else None
        result["labelled_but_not_evaluated_samples"] = labelled - evaluated
        result["status"] = (
            "labels_required" if not labelled else
            "predictions_unavailable" if not evaluated else
            "partially_evaluated" if evaluated < labelled else "evaluated"
        )
    summary["evaluation_status"] = readiness["status"]
    return summary


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Report already exists: {path}. Choose --report-name to preserve prior evaluation results.")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=default_test_directory(),
        help="Image directory (default: original uploads when available, otherwise test_images)",
    )
    parser.add_argument("--ground-truth", type=Path, help="JSON labels (default: <test-dir>/ground_truth.json)")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports", help="Directory for evaluation reports")
    parser.add_argument("--report-name", help="Optional report filename; must end in .json")
    parser.add_argument("--clip-threshold", type=float, default=0.30, help="Positive CLIP similarity cutoff in [-1, 1]")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N labelled samples")
    parser.add_argument("--check-labels", action="store_true", help="Show missing annotations without loading models")
    parser.add_argument("--allow-unlabelled", action="store_true", help="Run predictions even if no metric has ground-truth labels")
    parser.add_argument(
        "--create-ground-truth-template",
        action="store_true",
        help="Create a label-free ground_truth.json from images in --test-dir, then exit",
    )
    args = parser.parse_args()
    if not -1.0 <= args.clip_threshold <= 1.0:
        parser.error("--clip-threshold must be in [-1, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.report_name and Path(args.report_name).name != args.report_name:
        parser.error("--report-name must be a filename, not a path")
    if args.report_name and not args.report_name.lower().endswith(".json"):
        parser.error("--report-name must end in .json")
    return args


def main() -> int:
    args = parse_args()
    test_dir = args.test_dir.resolve()
    ground_truth_path = (args.ground_truth or test_dir / "ground_truth.json").resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test image directory was not found: {test_dir}")
    if args.create_ground_truth_template:
        sample_count = create_ground_truth_template(ground_truth_path, test_dir)
        print(
            f"Created label-free template: {ground_truth_path} ({sample_count} image records). "
            "Complete independent labels, then run the evaluator again.",
            flush=True,
        )
        return 0
    if not ground_truth_path.is_file():
        sample_count = create_ground_truth_template(ground_truth_path, test_dir)
        print(
            f"Ground truth was missing, so a label-free template was created at {ground_truth_path} "
            f"with {sample_count} image records. Complete its labels, then run the evaluator again.",
            flush=True,
        )
        return 0
    samples = load_ground_truth(ground_truth_path, test_dir)
    if args.limit:
        samples = samples[:args.limit]

    readiness = label_readiness(samples)
    print("Ground-truth label coverage:", flush=True)
    for item in readiness["metrics"].values():
        print(f"  {item['required_field']}: {item['labelled_samples']}/{len(samples)}", flush=True)
    if args.check_labels:
        print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)
        return 2 if readiness["status"] == "labels_required" else 0
    if readiness["status"] == "labels_required" and not args.allow_unlabelled:
        print(
            f"No accuracy labels are present in {ground_truth_path}. Models were not loaded. "
            "Add independent labels using docs/evaluation_system.md, or use --allow-unlabelled "
            "for predictions only (accuracy metrics will remain null).",
            flush=True,
        )
        return 2

    print(f"Evaluating {len(samples)} samples from {test_dir}", flush=True)
    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample['id']}", flush=True)
        results.append(run_pipeline(sample))

    timestamp = datetime.now(timezone.utc)
    report_name = args.report_name or f"evaluation_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path = args.report_dir.resolve() / report_name
    report = {
        "schema_version": "evaluation-report/v1",
        "created_at": timestamp.isoformat(),
        "test_images_directory": str(test_dir),
        "ground_truth_file": str(ground_truth_path),
        "pipeline": ["OCR", "translation", "CLIP cross-modal similarity", "AI authenticity detection", "local image reuse"],
        "clip_threshold": args.clip_threshold,
        "annotation_readiness": readiness,
        "annotations_pending_human_review": sum(
            sample.get("annotation_review_status") == "human_review_pending" for sample in samples
        ),
        "summary": build_summary(results, args.clip_threshold),
        "samples": results,
        "limitations": [
            "Assistant-provided annotations marked human_review_pending produce provisional metrics until reviewed by a human annotator.",
            "Metrics quantify this labelled test set only; they do not establish 100% accuracy or verify a real-world news claim.",
            "Image reuse evaluation is limited to data/reference_images configured in this project, not the public web.",
            "CLIP scores and authenticity scores are model indicators; their decision thresholds must be selected on a validation set, then confirmed on a separate held-out test set.",
        ],
    }
    write_report(report_path, report)
    print(json.dumps({"report": str(report_path), "summary": report["summary"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(2)

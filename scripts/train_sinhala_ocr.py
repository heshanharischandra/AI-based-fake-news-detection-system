"""Reproducible CPU Sinhala OCR fine-tuning on labeled synthetic news lines.

Never treats news headlines as verified facts. Keeps training, validation and
test headlines disjoint. Does not activate a candidate without validation gain.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/training/sinhala_ocr"
MODEL = ROOT / "models/sinhala_ocr"
REV = "1d07cd3b032c6ee077516d2f8a76633a171a236a"
DATASET = "edifier99/sinhala_synthetic_ocr_news_large"
BASE_REV = "e12c65a915945e4c28e237a9b52bc4a8f39a0cec"
PARQUET_SHA = "c0f2a85eccb83e90197e937b98e1fe38ede5b64db0bc693e4d50c180562119dd"
BASE_SHA = "1d95b36e935c8c12cc890684fad15bd3ad8d904c3a6ffa2fccb81658eabd8454"
DEFAULT_EVALUATION_PSM_MODES = (13,)
DEFAULT_RUNTIME_PSM_MODES = (6, 11)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def download(url, path, expected_sha=None, attempts=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_sha is None or sha256(path) == expected_sha):
        return
    temporary = path.with_suffix(path.suffix + ".download")
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {path.name} (attempt {attempt}/{attempts})", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "ImageVerificationSystem/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            if not temporary.stat().st_size:
                raise ValueError("Downloaded file is empty: " + str(path))
            if expected_sha and sha256(temporary) != expected_sha:
                raise ValueError("Download checksum mismatch: " + str(path))
            temporary.replace(path)
            return
        except (OSError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"Could not download verified file {path}: {error}") from error
            time.sleep(attempt)


def prepare(args):
    import pyarrow.parquet as pq

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REV}/data/train-00000-of-00001.parquet"
    parquet = DATA / "source.parquet"
    download(url, parquet, PARQUET_SHA)
    download(f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/{BASE_REV}/sin.traineddata",
             MODEL / "base/sin.traineddata", BASE_SHA)
    table = pq.read_table(parquet, columns=["image", "text"])
    # Repeated normalized headlines and identical images never cross splits.
    unique, seen_text, seen_image = [], set(), set()
    excluded = {"invalid_label": 0, "duplicate": 0}
    for index, row in enumerate(table.to_pylist()):
        label = normalize(row["text"])
        image_bytes = row["image"]["bytes"]
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if not any("\u0d80" <= c <= "\u0dff" and c.isalpha() for c in label) or "\ufffd" in label:
            excluded["invalid_label"] += 1
            continue
        key = label.casefold()
        if key in seen_text or image_hash in seen_image:
            excluded["duplicate"] += 1
            continue
        seen_text.add(key)
        seen_image.add(image_hash)
        unique.append((index, label, image_bytes))
    random.Random(20261018).shuffle(unique)
    required = args.train_samples + args.eval_samples * 2
    if required > len(unique):
        raise ValueError(f"Requested {required} unique lines, only {len(unique)} available")
    splits = {"train": unique[:args.train_samples],
              "validation": unique[args.train_samples:args.train_samples + args.eval_samples],
              "test": unique[args.train_samples + args.eval_samples:required]}
    manifest = {"dataset": DATASET, "revision": REV, "source_url": url,
                "license": "CC-BY-4.0 (dataset card)",
                "attribution": "Ransaka R., 2026, DOI 10.57967/hf/9748; edifier99 dataset",
                "domain": "synthetic Sinhala news text-line images; NOT real news screenshot validation",
                "source_sha256": sha256(parquet), "base_revision": BASE_REV,
                "base_sha256": sha256(MODEL / "base/sin.traineddata"),
                "seed": 20261018, "source_rows": table.num_rows, "unique_rows": len(unique),
                "excluded": excluded, "splits": {}}
    for split, rows in splits.items():
        destination = DATA / split
        destination.mkdir(parents=True, exist_ok=True)
        # Avoid silently retaining stale samples from a previous split size.
        selected_stems = {f"line_{index:05d}" for index, _, _ in rows}
        for stale in destination.glob("line_*.*"):
            if stale.stem.split(".")[0] not in selected_stems:
                stale.unlink()
        records = []
        for index, label, image_bytes in rows:
            stem = destination / f"line_{index:05d}"
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.convert("RGB").save(stem.with_suffix(".png"))
                width, height = image.size
            stem.with_suffix(".gt.txt").write_text(label + "\n", encoding="utf-8")
            # WordStr format preserves Indic combining marks and conjuncts.
            stem.with_suffix(".box").write_text(
                f"WordStr 0 0 {width} {height} 0 #{label}\n\t 0 0 {width} {height} 0\n",
                encoding="utf-8")
            records.append({"id": index, "text": label, "image": str(stem.with_suffix(".png")),
                            "source_image_sha256": hashlib.sha256(image_bytes).hexdigest()})
        manifest["splits"][split] = records
    write_json(DATA / "manifest.json", manifest)
    print("Prepared", {key: len(rows) for key, rows in splits.items()}, "from", len(unique), "unique lines", flush=True)


def run(command, timeout=180):
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    result = subprocess.run([str(c) for c in command], capture_output=True, encoding="utf-8",
                            errors="replace", timeout=timeout, env=env)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr[-4000:]}")
    return result


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def metrics(pairs):
    char_errors = word_errors = chars = words = exact = 0
    for reference, prediction in pairs:
        reference, prediction = normalize(reference), normalize(prediction)
        char_errors += distance(reference, prediction)
        word_errors += distance(reference.split(), prediction.split())
        chars += len(reference)
        words += len(reference.split())
        exact += reference == prediction
    return {"samples": len(pairs), "character_error_rate": char_errors / max(chars, 1),
            "word_error_rate": word_errors / max(words, 1),
            "exact_line_accuracy": exact / max(len(pairs), 1)}


def parse_tsv(output):
    """Return line-preserving text and character-weighted confidence from Tesseract TSV."""
    lines, confidences = {}, []
    for raw_line in output.splitlines()[1:]:
        columns = raw_line.split("\t", 11)
        if len(columns) != 12:
            continue
        word = columns[11].strip()
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        if not word or confidence < 0:
            continue
        key = tuple(columns[index] for index in (1, 2, 3, 4))
        lines.setdefault(key, []).append(word)
        confidences.append((confidence, len(word)))
    text = normalize("\n".join(" ".join(words) for words in lines.values()))
    weight = sum(length for _, length in confidences)
    confidence = sum(value * length for value, length in confidences) / max(weight, 1)
    return text, confidence


def _error_examples(pairs, limit=25):
    errors = []
    for reference, prediction, confidence, psm in pairs:
        reference_normalized, prediction_normalized = normalize(reference), normalize(prediction)
        if reference_normalized == prediction_normalized:
            continue
        error_count = distance(reference_normalized, prediction_normalized)
        errors.append({"reference": reference_normalized, "prediction": prediction_normalized,
                       "character_errors": error_count,
                       "normalized_error": error_count / max(len(reference_normalized), 1),
                       "ocr_confidence": round(confidence, 2), "psm": psm})
    errors.sort(key=lambda row: row["normalized_error"], reverse=True)
    return errors[:limit]


def load_benchmark(path):
    """Load user-labeled real screenshots without mixing them into synthetic training."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("Real benchmark manifest not found: " + str(manifest_path))
    if manifest_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = value.get("records", []) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("Real benchmark must contain a non-empty list of image/text records")
    records, seen = [], set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not row.get("image") or not row.get("text"):
            raise ValueError(f"Invalid real benchmark record {index}: image and text are required")
        image = Path(row["image"])
        if not image.is_absolute():
            image = manifest_path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Benchmark image does not exist: {image}")
        label = normalize(str(row["text"]))
        if not label or "\ufffd" in label:
            raise ValueError(f"Invalid UTF-8 benchmark label at record {index}")
        key = (str(image).casefold(), label.casefold())
        if key in seen:
            raise ValueError(f"Duplicate real benchmark record {index}: {image}")
        seen.add(key)
        records.append({"image": str(image), "text": label})
    return records


def evaluate(records, tessdata, binaries, psm_modes):
    """Evaluate isolated lines, not the full screenshot/headline runtime pipeline."""
    tsv_config = binaries / "tessdata/configs/tsv"
    if not tsv_config.is_file():
        raise FileNotFoundError("Tesseract TSV config is missing: " + str(tsv_config))

    def recognize(row):
        candidates = []
        for psm in psm_modes:
            result = run([binaries / "tesseract.exe", row["image"], "stdout", "--tessdata-dir", tessdata,
                          "-l", "sin", "--oem", "1", "--psm", str(psm), tsv_config])
            prediction, confidence = parse_tsv(result.stdout)
            if prediction:
                candidates.append((confidence, prediction, psm))
        confidence, prediction, psm = max(candidates, default=(0.0, "", psm_modes[0]))
        return row["text"], prediction, confidence, psm

    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(pool.map(recognize, records))
    summary = metrics([(reference, prediction) for reference, prediction, _, _ in pairs])
    psm_usage = {str(psm): sum(row[3] == psm for row in pairs) for psm in psm_modes}
    summary.update({"mean_ocr_confidence": sum(row[2] for row in pairs) / max(len(pairs), 1),
                    "psm_usage": psm_usage, "error_examples": _error_examples(pairs)})
    return summary


def audit_splits(records):
    """Refuse training if a source image or normalized headline crosses splits."""
    seen_text, seen_image = {}, {}
    for split in ("train", "validation", "test"):
        if not records.get(split):
            raise ValueError(f"Empty {split} split")
        for row in records[split]:
            label = normalize(row["text"]).casefold()
            if not label or "\ufffd" in label:
                raise ValueError(f"Invalid training label in {split}")
            image = Path(row["image"])
            digest = sha256(image)
            for key, seen in ((label, seen_text), (digest, seen_image)):
                if key in seen:
                    raise ValueError(f"Duplicate or overlapping samples: {seen[key]} / {split}")
                seen[key] = split
    return {"status": "passed", "samples": {k: len(v) for k, v in records.items()}}


def training_inputs_match(image, label):
    """The manifest is not ground truth if the actual training box disagrees."""
    ground_truth = image.with_suffix(".gt.txt")
    box = image.with_suffix(".box")
    if not all(p.is_file() and p.stat().st_size for p in (image, ground_truth, box)):
        return False
    if normalize(ground_truth.read_text(encoding="utf-8")) != normalize(label):
        return False
    box_labels = [line.split("#", 1)[1] for line in box.read_text(encoding="utf-8").splitlines()
                  if line.startswith("WordStr ") and "#" in line]
    return box_labels == [label]


def can_reuse_encoding(image, label):
    """Reuse only existing encodings with unchanged, matching label/box inputs."""
    ground_truth, box, encoded = (image.with_suffix(suffix) for suffix in (".gt.txt", ".box", ".lstmf"))
    if not training_inputs_match(image, label) or not encoded.is_file() or not encoded.stat().st_size:
        return False
    return encoded.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in (image, ground_truth, box))


def validate_resume_splits(checkpoint, records):
    """Do not resume weights on a reshuffled split that may expose old training samples as test."""
    for split in ("train", "validation"):
        listfile = checkpoint.parent / f"{split}.txt"
        if not listfile.is_file():
            raise ValueError(f"Resume checkpoint lacks its {split} provenance list: {listfile}")
        previous = {Path(line).resolve() for line in listfile.read_text(encoding="utf-8").splitlines() if line}
        current = {Path(row["image"]).resolve().with_suffix(".lstmf") for row in records[split]}
        if previous != current:
            raise ValueError(f"Resume {split} split differs from the checkpoint's original split")


def run_training(command, log_path, timeout=3600):
    """Keep a live log and heartbeat instead of hiding a long training run."""
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, OMP_THREAD_LIMIT="1")) as process:
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.monotonic() - started)
                    print(f"Training running: {elapsed}s; log: {log_path}", flush=True)
                    if elapsed >= timeout:
                        process.kill()
                        process.wait()
                        raise TimeoutError(f"Training timed out; checkpoint/log retained at {log_path.parent}")
            if code:
                raise RuntimeError(f"Training exited with {code}; inspect {log_path}")


def promotion_check(current_validation, candidate_validation, current_test, candidate_test,
                    minimum_gain=0.002, maximum_regression=0.005, real_regression=None):
    """Gate deployment without confusing lower CER with exact headline accuracy."""
    gain = current_validation["character_error_rate"] - candidate_validation["character_error_rate"]
    if gain < minimum_gain:
        return False, "validation improvement was below the required minimum"
    if candidate_test["character_error_rate"] - current_test["character_error_rate"] > maximum_regression:
        return False, "candidate regressed beyond the held-out test safety limit"
    if candidate_test["exact_line_accuracy"] < current_test["exact_line_accuracy"]:
        return False, "candidate regressed on exact-line accuracy"
    if candidate_test["word_error_rate"] > current_test["word_error_rate"]:
        return False, "candidate regressed on word error rate"
    if real_regression is not None and real_regression > maximum_regression:
        return False, "candidate regressed beyond the real-screenshot benchmark safety limit"
    return True, "candidate passed validation-gain and held-out-test safety gates"


def train(args):
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    binaries = Path(args.tesseract_dir)
    required_tools = ("tesseract.exe", "combine_tessdata.exe", "lstmtraining.exe")
    missing_tools = [name for name in required_tools if not (binaries / name).is_file()]
    if missing_tools:
        raise FileNotFoundError(f"Missing Tesseract training tools in {binaries}: {missing_tools}")
    base = MODEL / "base"
    run_name = getattr(args, "run_name", None)
    resume_from = getattr(args, "resume_from", None)
    if resume_from and not run_name:
        raise ValueError("--resume-from requires a new --run-name to preserve previous outputs")
    if run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("Invalid run name; use letters, digits, hyphens and underscores")
    run_root = MODEL / "runs" / run_name if run_name else MODEL
    if run_name:
        run_root.mkdir(parents=True, exist_ok=False)
    report_path = ROOT / "reports" / (f"sinhala_ocr_{run_name}.json" if run_name else "sinhala_ocr_training.json")
    work = run_root / "training"
    work.mkdir(parents=True, exist_ok=True)
    config = binaries / "tessdata/configs/lstm.train"
    records = manifest["splits"]
    split_audit = audit_splits(records)
    if manifest.get("base_sha256") != sha256(base / "sin.traineddata"):
        raise ValueError("Base model differs from the dataset's recorded training model")
    if resume_from:
        validate_resume_splits(Path(resume_from).resolve(), records)
    # Snapshot the deployed selection before any possible promotion.
    selection_path = MODEL / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    baseline = Path(selection.get("tessdata_dir", binaries / "tessdata"))
    if not (baseline / "sin.traineddata").is_file():
        raise FileNotFoundError("Selected baseline OCR model is missing: " + str(baseline))
    if run_name:
        write_json(run_root / "previous_selection.json", selection)
    baseline_sha = sha256(baseline / "sin.traineddata")
    for split in ("train", "validation"):
        def encode(row):
            image = Path(row["image"])
            if not training_inputs_match(image, row["text"]):
                raise ValueError(f"Ground-truth/box label differs from manifest: {image}")
            if getattr(args, "reuse_encoded", False) and can_reuse_encoding(image, row["text"]):
                return image.with_suffix(".lstmf").as_posix()
            run([binaries / "tesseract.exe", image, image.with_suffix(""),
                 "--tessdata-dir", base, "-l", "sin", "--psm", "13", config])
            path = image.with_suffix(".lstmf")
            if not path.exists() or not path.stat().st_size:
                raise RuntimeError("Training line not encoded: " + str(path))
            return path.as_posix()
        print("Encoding", split, len(records[split]), flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(encode, records[split]))
        # MinGW Tesseract's list reader does not reliably strip CR on Windows.
        (work / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
    run([binaries / "combine_tessdata.exe", "-e", base / "sin.traineddata", work / "base.lstm"])
    start_model = Path(resume_from).resolve() if resume_from else work / "base.lstm"
    if not start_model.is_file():
        raise FileNotFoundError("Resume checkpoint not found: " + str(start_model))
    print("Fine-tuning Sinhala LSTM with max_iterations =", args.iterations,
          "(checkpoint counters depend on the Tesseract build)", flush=True)
    run_training([binaries / "lstmtraining.exe", "--continue_from", start_model,
                  "--traineddata", base / "sin.traineddata", "--train_listfile", work / "train.txt",
                  "--eval_listfile", work / "validation.txt", "--model_output", work / "sin",
                  "--max_iterations", str(args.iterations), "--learning_rate", "0.0001",
                  "--target_error_rate", "0"], work / "training.log", timeout=3600)
    checkpoint = work / "sin_checkpoint"
    if not checkpoint.exists():
        raise RuntimeError("No training checkpoint produced; inspect training.log")
    candidate = run_root / "candidate"
    candidate.mkdir(exist_ok=True)
    run([binaries / "lstmtraining.exe", "--stop_training", "--continue_from", checkpoint,
         "--traineddata", base / "sin.traineddata", "--model_output", candidate / "sin.traineddata"])
    shutil.copy2(baseline / "eng.traineddata", candidate / "eng.traineddata")
    psm_modes = tuple(args.evaluation_psm_modes)
    print("Evaluating isolated-line validation set with PSM modes", psm_modes,
          "; test set remains unseen during selection", flush=True)
    sources = {"current": baseline, "fine_tuned": candidate}
    validation = {name: evaluate(records["validation"], path, binaries, psm_modes)
                  for name, path in sources.items()}
    winner = min(sources, key=lambda name: validation[name]["character_error_rate"])
    # Ties keep the deployed model. Compare against it, not a weaker stock model.
    print("Validation-selected model:", winner, "; evaluating held-out test", flush=True)
    test = {name: evaluate(records["test"], sources[name], binaries, psm_modes)
            for name in dict.fromkeys(["current", winner])}
    validation_gain = (validation["current"]["character_error_rate"]
                       - validation[winner]["character_error_rate"])
    test_regression = (test[winner]["character_error_rate"]
                       - test["current"]["character_error_rate"])
    real_benchmark = None
    real_regression = None
    if args.real_eval_manifest:
        from scripts.evaluate_ocr_runtime import evaluate_records, use_tessdata
        real_records = load_benchmark(args.real_eval_manifest)
        real_benchmark = {}
        for name in dict.fromkeys(["current", winner]):
            with use_tessdata(sources[name]):
                real_benchmark[name], _ = evaluate_records(real_records, psm_modes=tuple(args.runtime_psm_modes))
        real_regression = (real_benchmark[winner]["character_error_rate"]
                           - real_benchmark["current"]["character_error_rate"])
    if winner == "current":
        activation_allowed = False
        activation_reason = "current model had the lowest validation CER"
    else:
        activation_allowed, activation_reason = promotion_check(
            validation["current"], validation[winner], test["current"], test[winner],
            args.min_validation_gain, args.max_test_regression, real_regression)
    report = {"dataset": DATASET, "revision": REV, "domain": manifest["domain"],
              "baseline_sha256": baseline_sha, "candidate_sha256": sha256(candidate / "sin.traineddata"),
              "resume_from": str(start_model), "split_audit": split_audit,
              "manifest_sha256": sha256(DATA / "manifest.json"),
              "starting_checkpoint_sha256": sha256(start_model),
              "training_encoding_failures": (work / "training.log").read_text(encoding="utf-8", errors="replace").count("Can't encode transcription"),
              "iterations": args.iterations, "train_samples": len(records["train"]),
              "validation_samples": len(records["validation"]), "test_samples": len(records["test"]),
              "evaluation_psm_modes": list(psm_modes),
              "runtime_psm_modes": list(args.runtime_psm_modes),
              "model_selection": "minimum validation Unicode-codepoint CER on isolated lines; test not used to select",
              "selected_model": winner, "validation": validation, "test": test,
              "real_screenshot_benchmark": real_benchmark,
              "activation": {"requested": bool(args.activate), "allowed": activation_allowed,
                             "performed": bool(args.activate and activation_allowed),
                             "reason": activation_reason,
                             "minimum_validation_gain": args.min_validation_gain,
                             "observed_validation_gain": validation_gain,
                             "maximum_test_regression": args.max_test_regression,
                             "observed_test_regression": test_regression,
                             "observed_real_benchmark_regression": real_regression},
              "limitations": ["Synthetic single-line Sinhala images only, not real screenshots",
                              "Mixed-language OCR, translation and factual understanding not evaluated",
                              "CER measures Unicode codepoints after NFC; not grapheme clusters",
                              "No 100% accuracy guarantee"]}
    write_json(report_path, report)
    if args.activate and activation_allowed:
        if sha256(baseline / "sin.traineddata") != baseline_sha:
            raise RuntimeError("Baseline changed during training; refusing activation")
        active = run_root / "active"
        active.mkdir(exist_ok=True)
        shutil.copy2(sources[winner] / "sin.traineddata", active / "sin.traineddata")
        shutil.copy2(baseline / "eng.traineddata", active / "eng.traineddata")
        write_json(MODEL / "selection.json", {"selected_model": winner, "tessdata_dir": str(active),
                                              "sin_sha256": sha256(active / "sin.traineddata"),
                                              "evaluation_psm_modes": list(psm_modes),
                                              "runtime_psm_modes": list(args.runtime_psm_modes),
                                              "report": report_path.relative_to(ROOT).as_posix()})
    compact_test = {name: {k: v for k, v in result.items() if k != "error_examples"} for name, result in test.items()}
    print(json.dumps({"selected": winner, "activation": report["activation"], "test": compact_test},
                     ensure_ascii=False, indent=2), flush=True)


def parse_psm_modes(value):
    try:
        modes = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError("PSM modes must be comma-separated integers") from error
    if not modes or any(mode < 0 or mode > 13 for mode in modes):
        raise argparse.ArgumentTypeError("PSM modes must contain values from 0 through 13")
    return modes


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--eval-samples", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--run-name", help="New isolated run directory; existing runs are never overwritten")
    parser.add_argument("--resume-from", help="Continue from a checkpoint; iteration counter behaviour depends on Tesseract")
    parser.add_argument("--reuse-encoded", action="store_true", help="Reuse unchanged cached training line encodings")
    parser.add_argument("--evaluation-psm-modes", type=parse_psm_modes,
                        default=DEFAULT_EVALUATION_PSM_MODES,
                        help="PSM modes for isolated-line model evaluation (default: 13)")
    parser.add_argument("--runtime-psm-modes", type=parse_psm_modes,
                        default=DEFAULT_RUNTIME_PSM_MODES,
                        help="PSM modes saved for full-image runtime OCR (default: 6,11)")
    parser.add_argument("--real-eval-manifest",
                        help="Optional JSON/JSONL labeled real-screenshot benchmark with image/text records")
    parser.add_argument("--min-validation-gain", type=float, default=0.002,
                        help="Minimum absolute CER improvement required for activation")
    parser.add_argument("--max-test-regression", type=float, default=0.005,
                        help="Maximum held-out CER regression allowed for activation")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--tesseract-dir", default="C:/Program Files/Tesseract-OCR")
    args = parser.parse_args()
    if args.resume_from and args.stage != "train":
        parser.error("Resume with --stage train to preserve the original frozen dataset splits")
    if min(args.train_samples, args.eval_samples, args.iterations) < 1:
        parser.error("Sample counts and iterations must be positive")
    if args.min_validation_gain < 0 or args.max_test_regression < 0:
        parser.error("Activation gate values cannot be negative")
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("train", "all"):
        train(args)

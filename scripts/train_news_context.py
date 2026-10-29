"""Train a frozen-CLIP context head on official NewsCLIPpings splits.

Run from the project root: .venv311/Scripts/python scripts/train_news_context.py
No training data is inserted into the application's verified-evidence store.
"""
import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVISION = "a2bbe9a8125f47a5c26338125c71e05be0266cf7"
REPO = "g-luo/news-clippings"
DATA = ROOT / "data" / "training" / "newsclippings"
MODEL = ROOT / "models" / "news_context.json"
REPORT = ROOT / "reports" / "news_context_training.json"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    wanted = [f"data/merged_balanced/{split}.json" for split in ("train", "val", "test")]
    wanted += [f"embeddings/clip_{kind}_embeddings/clip_{kind}_embeddings_{split}.pkl"
               for split in ("train", "val", "test") for kind in ("image", "text")]
    manifest = {"dataset": REPO, "revision": REVISION, "files": [],
                "task": "original_vs_constructed_mismatched_news_pairs",
                "source": "https://github.com/g-luo/news_clippings",
                "not_verified_news_evidence": True}
    for remote in wanted:
        target = DATA / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = remote.rsplit("/", 1)[0]
        api = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}/{parent}"
        with urllib.request.urlopen(api, timeout=90) as response:
            metadata = next(item for item in json.load(response) if item["path"] == remote)
        size = metadata["size"]
        expected_sha = metadata.get("lfs", {}).get("oid")
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
        if target.exists() and target.stat().st_size == size:
            actual_sha = digest(target)
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(f"Checksum mismatch in cached file: {target}")
        else:
            print(f"Downloading {remote}: {size / 1e6:.1f} MB", flush=True)
            partial = target.with_suffix(target.suffix + ".partial")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
                        received = 0
                        last_update = time.monotonic()
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                            received += len(chunk)
                            if time.monotonic() - last_update > 20:
                                print(f"  {received / 1e6:.1f}/{size / 1e6:.1f} MB", flush=True)
                                last_update = time.monotonic()
                    if partial.stat().st_size != size:
                        raise ValueError("Downloaded size differs from repository metadata")
                    actual_sha = digest(partial)
                    if expected_sha and actual_sha != expected_sha:
                        raise ValueError("Downloaded SHA-256 differs from repository metadata")
                    partial.replace(target)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    print("Retrying incomplete download", flush=True)
        manifest["files"].append({"path": remote, "bytes": size, "sha256": actual_sha, "url": url})
        save_json(DATA / "manifest.json", manifest)


class ArrayUnpickler(pickle.Unpickler):
    """Only permit the numpy array constructors used by official features."""
    def find_class(self, module, name):
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            return getattr(np._core.multiarray, name)
        if module == "_codecs" and name == "encode":
            import codecs
            return codecs.encode
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")


def load_embeddings(split, kind):
    path = DATA / "embeddings" / f"clip_{kind}_embeddings" / f"clip_{kind}_embeddings_{split}.pkl"
    with path.open("rb") as stream:
        values = ArrayUnpickler(stream).load()
    if not isinstance(values, dict):
        raise ValueError("Expected a dictionary of CLIP embeddings")
    return values


def prepare():
    from app.services.news_context_classifier import pair_features
    id_sets = {}
    pair_sets = {}
    for split in ("train", "val", "test"):
        annotations = json.loads((DATA / "data" / "merged_balanced" / f"{split}.json").read_text())["annotations"]
        id_sets[split] = {int(row[key]) for row in annotations for key in ("id", "image_id")}
        pair_sets[split] = {(row["id"], row["image_id"]) for row in annotations}
        if len(pair_sets[split]) != len(annotations):
            raise ValueError(f"Repeated image/caption pairs in {split}")
        print(f"Preparing {split}: {len(annotations)} pairs", flush=True)
        image_lookup = load_embeddings(split, "image")
        text_lookup = load_embeddings(split, "text")
        image = np.stack([np.asarray(image_lookup[row["image_id"]]).reshape(-1) for row in annotations])
        text = np.stack([np.asarray(text_lookup[row["id"]]).reshape(-1) for row in annotations])
        del image_lookup, text_lookup
        gc.collect()
        labels = []
        for row in annotations:
            if type(row["falsified"]) is not bool:
                raise ValueError("Expected boolean falsified labels")
            labels.append(int(row["falsified"]))
        features = pair_features(image, text)
        np.savez(DATA / f"{split}.npz", x=features, y=np.array(labels, dtype=np.int64))
        del features, image, text
        gc.collect()
    audit = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = id_sets[first] & id_sets[second]
        audit[f"{first}_{second}_shared_visualnews_ids"] = len(shared)
        if shared:
            raise ValueError(f"Cross-split source ID overlap: {first}/{second}; refusing training")
    save_json(DATA / "split_audit.json", audit)


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, brier_score_loss
    from threadpoolctl import threadpool_limits
    arrays = {}
    for split in ("train", "val", "test"):
        with np.load(DATA / f"{split}.npz", allow_pickle=False) as packed:
            arrays[split] = (packed["x"], packed["y"])
    if not (DATA / "split_audit.json").exists():
        raise ValueError("Run preparation and split-leakage audit first")
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    best = None
    candidates = []
    with threadpool_limits(limits=4):
        for strength in (0.01, 0.1, 1.0, 10.0):
            print(f"Training regularized logistic context head C={strength}", flush=True)
            estimator = LogisticRegression(C=strength, max_iter=600, solver="lbfgs", random_state=42)
            estimator.fit(x_train, y_train)
            validation_scores = estimator.predict_proba(x_val)[:, 1]
            for threshold in np.linspace(0.3, 0.7, 41):
                value = balanced_accuracy_score(y_val, validation_scores >= threshold)
                if best is None or value > best[0]:
                    best = (value, estimator, float(threshold), strength)
            candidates.append({"C": strength, "validation_auc": float(roc_auc_score(y_val, validation_scores)),
                               "iterations": int(estimator.n_iter_[0])})
    _, estimator, threshold, strength = best

    def metrics(x, y):
        scores = estimator.predict_proba(x)[:, 1]
        predicted = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
        return {"samples": len(y), "constructed_mismatch_count": int(y.sum()),
                "accuracy": float(accuracy_score(y, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                "precision_mismatch": float(precision), "recall_mismatch": float(recall), "f1_mismatch": float(f1),
                "roc_auc": float(roc_auc_score(y, scores)), "brier_score": float(brier_score_loss(y, scores)),
                "confusion_matrix_labels_0_original_1_mismatch": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}

    artifact = {"schema_version": 1, "dataset": "NewsCLIPpings merged_balanced", "dataset_revision": REVISION,
                "encoder": "openai/clip-vit-base-patch32", "text_mode": "raw_caption_no_prompt",
                "feature_version": "normalized_product_absdiff_cosine_v1", "embedding_dimension": 512,
                "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
                "threshold": threshold, "regularization_C": strength,
                "calibration_status": "not_probability_calibrated", "encoder_finetuned": False,
                "label_0": "original_dataset_pair", "label_1": "constructed_out_of_context_pair"}
    report = {"dataset": artifact["dataset"], "revision": REVISION, "train_samples": len(y_train),
              "validation": metrics(*arrays["val"]), "test": metrics(*arrays["test"]),
              "selection": {"metric": "validation_balanced_accuracy", "C": strength, "threshold": threshold},
              "candidates": candidates, "split_audit": json.loads((DATA / "split_audit.json").read_text()),
              "limitations": ["Frozen CLIP encoder; only the classification head was trained.",
                              "Constructed mismatches are not comprehensive real-world misinformation labels.",
                              "English news benchmark; Sinhala OCR and translated headlines are not validated.",
                              "Does not detect AI generation or independently verify news truth."]}
    # Compare with a cosine-only rule selected on validation, not the test set.
    cosine_cutoffs = np.linspace(float(x_val[:, -1].min()), float(x_val[:, -1].max()), 101)
    cutoff = max(cosine_cutoffs, key=lambda value: balanced_accuracy_score(y_val, x_val[:, -1] < value))
    x_test, y_test = arrays["test"]
    report["cosine_only_baseline"] = {
        "validation_selected_cutoff": float(cutoff),
        "test_accuracy": float(accuracy_score(y_test, x_test[:, -1] < cutoff)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, x_test[:, -1] < cutoff)),
    }
    exported_scores = 1 / (1 + np.exp(-np.clip(x_test @ np.asarray(artifact["coef"]) + artifact["intercept"], -60, 60)))
    if not np.allclose(exported_scores, estimator.predict_proba(x_test)[:, 1], atol=1e-6):
        raise ValueError("Exported model does not reproduce classifier predictions")
    report["export_parity_passed"] = True
    save_json(MODEL, artifact)
    save_json(REPORT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "download", "prepare", "train"), default="all")
    arguments = parser.parse_args()
    if arguments.stage in ("all", "download"):
        download()
    if arguments.stage in ("all", "prepare"):
        prepare()
    if arguments.stage in ("all", "train"):
        train()

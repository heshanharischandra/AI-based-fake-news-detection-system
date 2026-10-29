
import csv
import json

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from services.identity_verification import analyze_identity_candidates


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "identity_pairs_reviewed.csv"

RESULTS_DIR = BASE_DIR / "results"

THRESHOLD = 0.75


def evaluate():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Complete the manual review first. "
            "Expected data/identity_pairs_reviewed.csv"
        )

    result = analyze_identity_candidates(
        DATA_PATH
    )

    pairs = result["pairs"]

    rows = []
    y_true = []
    y_pred = []

    for pair in pairs:
        similarity = pair[
            "features"
        ]["name_similarity"]

        label = pair[
            "review"
        ]["identity_label"]

        review_status = pair[
            "review"
        ]["status"]

        prediction = (
            1
            if similarity >= THRESHOLD
            else 0
        )

        is_labeled = (
            review_status == "reviewed"
            and label in {
                "same_publisher",
                "different_publisher"
            }
        )

        if is_labeled:
            truth = (
                1
                if label == "same_publisher"
                else 0
            )

            y_true.append(truth)
            y_pred.append(prediction)

        rows.append({
            "publisher_id": pair["publisher_id"],
            "publisher_name": pair["publisher_name"],
            "platform": pair["platform"],
            "account_url": pair["account_url"],
            "name_similarity": similarity,
            "baseline_candidate_match": prediction,
            "human_label": label,
            "review_status": review_status,
            "included_in_evaluation": is_labeled
        })

    positives = sum(y_true)
    negatives = len(y_true) - positives

    summary = {
        "experiment": "name_similarity_baseline",
        "threshold": THRESHOLD,
        "threshold_selection": "predefined_not_tuned",
        "total_pairs": len(pairs),
        "reviewed_labeled_pairs": len(y_true),
        "positive_labels": positives,
        "negative_labels": negatives,
        "unlabeled_or_uncertain": (
            len(pairs) - len(y_true)
        ),
        "prediction_meaning": (
            "Candidate match based on account-handle "
            "name similarity, not verified ownership"
        )
    }

    if positives >= 3 and negatives >= 3:
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        summary["evaluation_status"] = (
            "exploratory_small_sample"
        )

        summary["metrics"] = {
            "accuracy": round(
                accuracy_score(
                    y_true,
                    y_pred
                ),
                4
            ),
            "balanced_accuracy": round(
                balanced_accuracy_score(
                    y_true,
                    y_pred
                ),
                4
            ),
            "precision": round(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),
            "recall": round(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),
            "f1_score": round(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn)
        }

    else:
        summary["evaluation_status"] = (
            "insufficient_reviewed_labels"
        )

        summary["required_next_action"] = (
            "Review more independently established "
            "same-publisher and different-publisher pairs"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_path = (
        RESULTS_DIR
        / "identity_baseline_predictions.csv"
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ) if rows else [
                "publisher_id",
                "publisher_name",
                "platform",
                "account_url",
                "name_similarity",
                "baseline_candidate_match",
                "human_label",
                "review_status",
                "included_in_evaluation"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    metrics_path = (
        RESULTS_DIR
        / "identity_baseline_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print()
    print("Prediction file:", predictions_path)
    print("Evaluation file:", metrics_path)


if __name__ == "__main__":
    evaluate()

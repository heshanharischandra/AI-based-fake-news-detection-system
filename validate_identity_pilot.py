
import csv
import json
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_identity_pilot import (
    FEATURES,
    LABEL_MAP,
    extract_features,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "identity_pairs_reviewed.csv"
RESULTS = ROOT / "results"


def evaluate():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA}"
        )

    df = pd.read_csv(
        DATA,
        dtype=str
    ).fillna("")

    required = {
        "publisher_id",
        "publisher_name",
        "website_url",
        "account_url",
        "platform",
        "identity_label",
        "review_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["identity_label"] = (
        df["identity_label"]
        .str.strip()
        .str.lower()
    )

    df["review_status"] = (
        df["review_status"]
        .str.strip()
        .str.lower()
    )

    df = df[
        (df["review_status"] == "reviewed")
        & df["identity_label"].isin(LABEL_MAP)
    ].copy()

    df["publisher_id"] = (
        df["publisher_id"]
        .str.strip()
        .str.upper()
    )

    df["account_url"] = (
        df["account_url"]
        .str.strip()
        .str.rstrip("/")
        .str.lower()
    )

    keys = [
        "publisher_id",
        "platform",
        "account_url",
    ]

    conflicting = (
        df.groupby(keys)["identity_label"]
        .nunique()
    )

    if (conflicting > 1).any():
        raise ValueError(
            "Conflicting labels found. "
            "Review the dataset before evaluation."
        )

    df = (
        df.drop_duplicates(subset=keys)
        .reset_index(drop=True)
    )

    if (
        df["publisher_id"].nunique() < 3
        or df["identity_label"].nunique() < 2
    ):
        raise ValueError(
            "Need at least three publishers "
            "and both reviewed label classes."
        )

    X = pd.DataFrame([
        extract_features(row)
        for _, row in df.iterrows()
    ])[FEATURES]

    y = (
        df["identity_label"]
        .map(LABEL_MAP)
        .astype(int)
    )

    groups = df["publisher_id"]

    predictions = []
    skipped = []

    splitter = LeaveOneGroupOut()

    for train_idx, test_idx in splitter.split(
        X,
        y,
        groups
    ):
        held_out_publisher = str(
            groups.iloc[test_idx[0]]
        )

        if y.iloc[train_idx].nunique() < 2:
            skipped.append(
                held_out_publisher
            )
            continue

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predicted = model.predict(
            X.iloc[test_idx]
        )

        training_accounts = set(
            df.iloc[train_idx]["account_url"]
        )

        for position, prediction in zip(
            test_idx,
            predicted
        ):
            row = df.iloc[position]

            predictions.append({
                "publisher_id": row["publisher_id"],
                "publisher_name": row["publisher_name"],
                "platform": row["platform"],
                "account_url": row["account_url"],
                "actual": int(y.iloc[position]),
                "logistic_prediction": int(prediction),
                "baseline_prediction": int(
                    float(
                        X.iloc[position][
                            "name_handle_similarity"
                        ]
                    ) >= 0.75
                ),
                "social_account_seen_in_other_train_group": bool(
                    row["account_url"] in training_accounts
                ),
            })

    if not predictions:
        raise ValueError(
            "No evaluation folds completed. "
            "More reviewed data is required."
        )

    actual = [
        row["actual"]
        for row in predictions
    ]

    def metrics(column):
        predicted = [
            row[column]
            for row in predictions
        ]

        return {
            "accuracy": round(
                accuracy_score(actual, predicted),
                4
            ),
            "balanced_accuracy": round(
                balanced_accuracy_score(
                    actual,
                    predicted
                ),
                4
            ),
            "precision": round(
                precision_score(
                    actual,
                    predicted,
                    zero_division=0
                ),
                4
            ),
            "recall": round(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0
                ),
                4
            ),
            "f1": round(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0
                ),
                4
            ),
            "confusion_matrix": confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            ).tolist(),
        }

    report = {
        "validation": "exploratory_leave_one_publisher_out",
        "reviewed_labeled_pairs": len(df),
        "tested_pairs": len(predictions),
        "tested_publishers": len({
            row["publisher_id"]
            for row in predictions
        }),
        "skipped_publisher_groups": skipped,
        "pairs_with_account_leakage": sum(
            row["social_account_seen_in_other_train_group"]
            for row in predictions
        ),
        "logistic_regression": metrics(
            "logistic_prediction"
        ),
        "name_similarity_baseline": metrics(
            "baseline_prediction"
        ),
        "limitation": (
            "Exploratory small-sample validation. "
            "Shared accounts between training and test "
            "groups and similarity-derived manual labels "
            "can bias evaluation results."
        ),
    }

    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        RESULTS
        / "identity_grouped_validation.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    csv_path = (
        RESULTS
        / "identity_grouped_predictions.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(predictions[0])
        )

        writer.writeheader()
        writer.writerows(predictions)

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        )
    )

    print("\nValidation report:", report_path)
    print("Prediction records:", csv_path)


if __name__ == "__main__":
    evaluate()

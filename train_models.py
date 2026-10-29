
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score
)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from services.analysis import _WEIGHTS


INPUT = Path(
    "data/reviewed_publishers.csv"
)

OUTPUT = Path(
    "models"
)

FEATURES = list(
    _WEIGHTS
)


def train():
    if not INPUT.exists():
        raise FileNotFoundError(
            "Create data/reviewed_publishers.csv "
            "using independent reviewer labels"
        )

    data = pd.read_csv(
        INPUT
    )

    required = [
        "domain",
        "reviewed_label",
        *FEATURES
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data.dropna(
        subset=[
            "domain",
            "reviewed_label"
        ]
    ).copy()

    data["reviewed_label"] = pd.to_numeric(
        data["reviewed_label"],
        errors="raise"
    ).astype(int)

    if (
        len(set(data["reviewed_label"])) != 2
        or not set(
            data["reviewed_label"]
        ).issubset({0, 1})
        or data["domain"].nunique() < 30
    ):
        raise ValueError(
            "Need at least 30 different publishers "
            "and two independently reviewed classes"
        )

    X = data[FEATURES].replace({
        "observed": 1,
        "not_established": pd.NA,
        "declared_not_independently_verified": 1,
        "not_found": 0
    }).apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = data[
        "reviewed_label"
    ]

    groups = (
        data["domain"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    if (
        y.iloc[train_idx].nunique() != 2
        or y.iloc[test_idx].nunique() != 2
    ):
        raise ValueError(
            "Both classes must exist in "
            "train and test publishers"
        )

    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(
                strategy="constant",
                fill_value=-1
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced"
            )
        ),
        "random_forest": make_pipeline(
            SimpleImputer(
                strategy="constant",
                fill_value=-1
            ),
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(
                strategy="constant",
                fill_value=-1
            ),
            HistGradientBoostingClassifier(
                random_state=42
            )
        )
    }

    OUTPUT.mkdir(
        exist_ok=True
    )

    report = {
        "label_definition": "independent_human_review",
        "train_publishers": int(
            groups.iloc[train_idx].nunique()
        ),
        "test_publishers": int(
            groups.iloc[test_idx].nunique()
        ),
        "results": {}
    }

    for name, model in models.items():
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        prediction = model.predict(
            X.iloc[test_idx]
        )

        report["results"][name] = {
            "balanced_accuracy": round(
                float(
                    balanced_accuracy_score(
                        y.iloc[test_idx],
                        prediction
                    )
                ),
                4
            ),
            "macro_f1": round(
                float(
                    f1_score(
                        y.iloc[test_idx],
                        prediction,
                        average="macro",
                        zero_division=0
                    )
                ),
                4
            ),
            "classification_report": classification_report(
                y.iloc[test_idx],
                prediction,
                output_dict=True,
                zero_division=0
            )
        }

        joblib.dump(
            {
                "pipeline": model,
                "feature_names": FEATURES,
                "label_definition": "independent_human_review"
            },
            OUTPUT / f"{name}.joblib"
        )

    (
        OUTPUT / "evaluation.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    summary = {
        name: {
            "balanced_accuracy": metrics[
                "balanced_accuracy"
            ],
            "macro_f1": metrics[
                "macro_f1"
            ]
        }
        for name, metrics in report[
            "results"
        ].items()
    }

    print(
        json.dumps(
            summary,
            indent=2
        )
    )


if __name__ == "__main__":
    train()

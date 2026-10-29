
from pathlib import Path
from urllib.parse import urlsplit
import csv

import joblib
import pandas as pd

# Reuse EXACTLY the same features as training.
from train_identity_pilot import FEATURES, extract_features


# ============================================================
# STEP 1 - PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "models" / "identity_matcher_pilot.joblib"
)

TRAINING_CSV = (
    ROOT / "results" / "identity_model_training_data.csv"
)

FACEBOOK_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
}

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
}


# ============================================================
# STEP 2 - VALIDATE PUBLISHER WEBSITE
# ============================================================

def validate_website(url):

    parsed = urlsplit(url)

    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "Enter a valid publisher website URL "
            "starting with https:// or http://"
        )


# ============================================================
# STEP 3 - VALIDATE SOCIAL ACCOUNT URL
# ============================================================

def validate_account(url):

    parsed = urlsplit(url)

    host = (parsed.hostname or "").lower()

    path = parsed.path.strip("/")

    if (
        parsed.scheme not in {"https", "http"}
        or not path
    ):
        raise ValueError(
            "Enter a complete Facebook or YouTube account URL."
        )

    if parsed.username or parsed.password:
        raise ValueError(
            "Account URLs must not contain credentials."
        )

    if host in FACEBOOK_HOSTS:

        if (
            path.isdigit()
            or path.lower().startswith("profile.php")
            or "/posts/" in "/" + path.lower() + "/"
            or path.lower().startswith(
                ("groups/", "reel/", "watch")
            )
        ):
            raise ValueError(
                "Use a named Facebook Page URL, "
                "not a numeric ID or post URL. "
                "This pilot relies on account-handle text."
            )

        return "facebook"

    if host in YOUTUBE_HOSTS:

        if not (
            path.startswith("@")
            or path.startswith(
                ("c/", "channel/", "user/")
            )
        ):
            raise ValueError(
                "Enter a YouTube channel URL, "
                "not a video URL."
            )

        return "youtube"

    raise ValueError(
        "Only facebook.com and youtube.com "
        "account URLs are supported."
    )


# ============================================================
# STEP 4 - CHECK IF PAIR WAS USED IN TRAINING
# ============================================================

def normalized_account(url):

    parsed = urlsplit(url)

    host = (
        (parsed.hostname or "")
        .lower()
        .removeprefix("www.")
    )

    path = parsed.path.rstrip("/").casefold()

    return host, path


def seen_in_training(publisher_name, account_url):

    if not TRAINING_CSV.exists():
        return None

    target_account = normalized_account(
        account_url
    )

    with TRAINING_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        for row in csv.DictReader(file):

            same_name = (
                row.get("publisher_name", "")
                .strip()
                .casefold()
                ==
                publisher_name.strip().casefold()
            )

            same_account = (
                normalized_account(
                    row.get("account_url", "")
                )
                ==
                target_account
            )

            if same_name and same_account:
                return True

    return False


# ============================================================
# STEP 5 - LOAD THE TRAINED MODEL
# ============================================================

def load_identity_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Load only your own locally trained model file.
    saved = joblib.load(MODEL_PATH)

    if (
        not isinstance(saved, dict)
        or "model" not in saved
    ):
        raise ValueError(
            "Expected model dictionary created by "
            "train_identity_pilot.py"
        )

    model = saved["model"]

    feature_columns = saved.get(
        "feature_columns"
    )

    if (
        not hasattr(model, "predict")
        or not isinstance(feature_columns, list)
    ):
        raise ValueError(
            "Saved model or feature metadata is invalid."
        )

    if set(feature_columns) != set(FEATURES):
        raise ValueError(
            "Saved model features do not match "
            "train_identity_pilot.py"
        )

    label_map = saved["label_map"]

    reverse_labels = {
        int(number): label
        for label, number in label_map.items()
    }

    return (
        model,
        feature_columns,
        reverse_labels,
        saved.get("prototype_only", True),
    )


# ============================================================
# STEP 6 - MAKE AN EXPERIMENTAL PREDICTION
# ============================================================

def predict_identity(
    model,
    feature_columns,
    reverse_labels,
    publisher_name,
    website_url,
    account_url,
):

    validate_website(website_url)

    platform = validate_account(
        account_url
    )

    row = {
        "publisher_name": publisher_name,
        "website_url": website_url,
        "account_url": account_url,
    }

    # Same feature engineering used in training.
    features = extract_features(row)

    input_data = pd.DataFrame(
        [features],
        columns=feature_columns
    )

    prediction_number = int(
        model.predict(input_data)[0]
    )

    label = reverse_labels.get(
        prediction_number,
        f"class_{prediction_number}"
    )

    return platform, label, features


# ============================================================
# STEP 7 - INTERACTIVE TERMINAL DEMO
# ============================================================

def main():

    print("=" * 65)
    print("CROSS-PLATFORM NEWS SOURCE IDENTITY MATCHING")
    print("LOGISTIC REGRESSION - EXPERIMENTAL PILOT")
    print("=" * 65)

    (
        model,
        feature_columns,
        reverse_labels,
        prototype_only,
    ) = load_identity_model()

    print("\nMODEL LOADED SUCCESSFULLY")

    print("Model type:", type(model).__name__)

    print("Prototype only:", prototype_only)

    print("\nEnter EXIT to stop the program.")

    while True:

        print("\n" + "=" * 65)

        publisher_name = input(
            "Publisher name: "
        ).strip()

        if publisher_name.upper() == "EXIT":
            break

        if not publisher_name:
            print("Publisher name cannot be empty.")
            continue

        website_url = input(
            "Publisher website URL: "
        ).strip()

        if website_url.upper() == "EXIT":
            break

        account_url = input(
            "Facebook/YouTube account URL: "
        ).strip()

        if account_url.upper() == "EXIT":
            break

        try:

            (
                platform,
                prediction,
                features,
            ) = predict_identity(
                model,
                feature_columns,
                reverse_labels,
                publisher_name,
                website_url,
                account_url,
            )

            print("\n" + "-" * 65)
            print("IDENTITY VERIFICATION - PILOT RESULT")
            print("-" * 65)

            print("Publisher:", publisher_name)
            print("Website:", website_url)
            print("Platform:", platform)
            print("Account:", account_url)

            print("\nEXPERIMENTAL PREDICTION:")
            print(prediction)

            print("\nMODEL INPUT FEATURES:")

            for feature_name in feature_columns:

                print(
                    f"  {feature_name}: "
                    f"{features[feature_name]}"
                )

            already_seen = seen_in_training(
                publisher_name,
                account_url
            )

            print("\nTRAINING DATA CHECK:")

            if already_seen is True:

                print(
                    "WARNING: This exact pair was "
                    "used during model training."
                )

            elif already_seen is False:

                print(
                    "This exact pair was not found "
                    "in the saved training CSV."
                )

                print(
                    "However, its publisher or account "
                    "may still overlap with training data."
                )

            else:

                print(
                    "Training CSV unavailable for "
                    "comparison."
                )

            print("\nIMPORTANT:")

            print(
                "This prediction is experimental."
            )

            print(
                "It does NOT confirm account ownership."
            )

            print(
                "The model has not been independently "
                "validated."
            )

        except (ValueError, KeyError) as error:

            print("\nINPUT / DATA ERROR:")
            print(error)

    print("\nProgram exited.")
    print("No models or CSV files were modified.")


if __name__ == "__main__":
    main()

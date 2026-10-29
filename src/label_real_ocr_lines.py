from pathlib import Path
import unicodedata

import cv2
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

CSV_PATH = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels.csv"
)


# ============================================================
# SETTINGS
# ============================================================

DISPLAY_SCALE = 3

MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 500


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text)

    text = text.replace(
        "\ufeff",
        ""
    )

    text = unicodedata.normalize(
        "NFC",
        text
    )

    return text.strip()


# ============================================================
# DISPLAY IMAGE
# ============================================================

def prepare_display(image):

    height, width = image.shape[:2]

    new_width = (
        width * DISPLAY_SCALE
    )

    new_height = (
        height * DISPLAY_SCALE
    )

    scale = min(
        DISPLAY_SCALE,
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height
    )

    scale = max(
        1.0,
        scale
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    enlarged = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return enlarged


# ============================================================
# SAVE CSV
# ============================================================

def save_dataframe(df):

    df.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL SINHALA OCR GROUND-TRUTH LABELING"
    )

    print(
        "=" * 72
    )

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in df.columns
    ]

    if "ground_truth" not in df.columns:

        df["ground_truth"] = ""

    if "keep" not in df.columns:

        df["keep"] = "1"

    df["ground_truth"] = (
        df["ground_truth"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Current progress
    # --------------------------------------------------------

    already_labeled = (
        df["ground_truth"]
        .str.strip()
        .ne("")
    )

    labeled_count = int(
        already_labeled.sum()
    )

    print(
        f"\nTotal samples   : "
        f"{len(df)}"
    )

    print(
        f"Already labeled : "
        f"{labeled_count}"
    )

    print(
        f"Remaining       : "
        f"{len(df) - labeled_count}"
    )

    print(
        "\nCommands:"
    )

    print(
        "  Type exact visible text = save label"
    )

    print(
        "  /skip  = leave for later"
    )

    print(
        "  /bad   = reject unusable crop"
    )

    print(
        "  /quit  = save progress and stop"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Type exactly what is visible in the image."
    )

    print(
        "Keep punctuation, numbers, and English "
        "characters exactly as shown."
    )

    # ========================================================
    # LABEL LOOP
    # ========================================================

    for index in range(
        len(df)
    ):

        current_ground_truth = (
            normalize_text(
                df.at[
                    index,
                    "ground_truth"
                ]
            )
        )

        # Resume support
        if current_ground_truth:

            continue

        sample_id = str(
            df.at[
                index,
                "sample_id"
            ]
        )

        relative_path = str(
            df.at[
                index,
                "image_path"
            ]
        )

        image_path = (
            OCR_ROOT
            / relative_path
        )

        if not image_path.exists():

            print(
                f"\nMissing image: "
                f"{image_path}"
            )

            df.at[
                index,
                "keep"
            ] = 0

            save_dataframe(
                df
            )

            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"\nCould not read: "
                f"{image_path}"
            )

            df.at[
                index,
                "keep"
            ] = 0

            save_dataframe(
                df
            )

            continue

        display_image = (
            prepare_display(
                image
            )
        )

        window_name = (
            f"{sample_id} "
            f"- read this line"
        )

        cv2.imshow(
            window_name,
            display_image
        )

        # Force OpenCV window to display
        cv2.waitKey(100)

        print(
            "\n" + "-" * 72
        )

        print(
            f"Sample: "
            f"{sample_id}"
        )

        print(
            f"Progress: "
            f"{index + 1}/"
            f"{len(df)}"
        )

        user_input = input(
            "Ground truth > "
        )

        user_input = (
            user_input.strip()
        )

        cv2.destroyWindow(
            window_name
        )

        # ----------------------------------------------------
        # Commands
        # ----------------------------------------------------

        if user_input.lower() == "/quit":

            save_dataframe(
                df
            )

            print(
                "\nProgress saved."
            )

            print(
                "Labeling stopped."
            )

            break

        if user_input.lower() == "/skip":

            print(
                "Skipped for now."
            )

            continue

        if user_input.lower() == "/bad":

            df.at[
                index,
                "keep"
            ] = 0

            df.at[
                index,
                "ground_truth"
            ] = ""

            save_dataframe(
                df
            )

            print(
                "Marked as unusable."
            )

            continue

        # Empty input
        if not user_input:

            print(
                "Empty label - skipped."
            )

            continue

        # ----------------------------------------------------
        # Save ground truth
        # ----------------------------------------------------

        ground_truth = (
            normalize_text(
                user_input
            )
        )

        df.at[
            index,
            "ground_truth"
        ] = ground_truth

        df.at[
            index,
            "keep"
        ] = 1

        # Save after every label
        save_dataframe(
            df
        )

        print(
            "Saved."
        )

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig"
    )

    df["ground_truth"] = (
        df["ground_truth"]
        .fillna("")
        .astype(str)
    )

    labeled = (
        df["ground_truth"]
        .str.strip()
        .ne("")
    )

    keep_values = (
        df["keep"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    rejected = (
        keep_values
        == "0"
    )

    usable_labeled = (
        labeled
        & ~rejected
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "LABELING STATUS"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTotal samples    : "
        f"{len(df)}"
    )

    print(
        f"Labeled          : "
        f"{int(labeled.sum())}"
    )

    print(
        f"Rejected         : "
        f"{int(rejected.sum())}"
    )

    print(
        f"Usable labeled   : "
        f"{int(usable_labeled.sum())}"
    )

    print(
        f"Still unlabeled  : "
        f"{int((~labeled & ~rejected).sum())}"
    )

    print(
        "\nCSV:"
    )

    print(
        CSV_PATH
    )
from pathlib import Path
import shutil

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

ACCEPTED_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "auto_line_candidates_accepted.csv"
)

REVIEWED_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "auto_line_labels_verified.csv"
)

BACKUP_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "auto_line_labels_verified_backup.csv"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

MAX_DISPLAY_WIDTH = 1400
MIN_SCALE = 2.0
MAX_SCALE = 4.0


# ============================================================
# LOAD / INITIALIZE DATA
# ============================================================

def load_review_data():

    if REVIEWED_CSV.exists():

        df = pd.read_csv(
            REVIEWED_CSV,
            encoding="utf-8-sig"
        )

        print(
            "Resuming existing review file."
        )

        shutil.copy2(
            REVIEWED_CSV,
            BACKUP_CSV
        )

        print(
            "Backup created:"
        )

        print(
            BACKUP_CSV
        )

    else:

        if not ACCEPTED_CSV.exists():

            raise FileNotFoundError(
                f"Accepted candidate CSV not found:\n"
                f"{ACCEPTED_CSV}"
            )

        df = pd.read_csv(
            ACCEPTED_CSV,
            encoding="utf-8-sig"
        )

        df["ground_truth"] = ""

        df["keep"] = 0

        df["review_status"] = "unreviewed"

        df.to_csv(
            REVIEWED_CSV,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "Created new review file:"
        )

        print(
            REVIEWED_CSV
        )

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in df.columns
    ]

    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""

    if "keep" not in df.columns:
        df["keep"] = 0

    if "review_status" not in df.columns:
        df["review_status"] = "unreviewed"

    df["ground_truth"] = (
        df["ground_truth"]
        .fillna("")
        .astype(str)
    )

    df["review_status"] = (
        df["review_status"]
        .fillna("unreviewed")
        .astype(str)
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save_dataframe(df):

    df.to_csv(
        REVIEWED_CSV,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# DISPLAY IMAGE
# ============================================================

def prepare_display(image):

    height, width = (
        image.shape[:2]
    )

    scale = max(
        MIN_SCALE,
        700 / max(
            width,
            1
        )
    )

    scale = min(
        scale,
        MAX_SCALE
    )

    new_width = int(
        width * scale
    )

    if new_width > MAX_DISPLAY_WIDTH:

        scale = (
            MAX_DISPLAY_WIDTH
            / width
        )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    display = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return display


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "VERIFIED LABELING - AUTO EXTRACTED REAL OCR LINES"
    )

    print(
        "=" * 72
    )

    df = load_review_data()

    total_samples = len(
        df
    )

    print(
        f"\nTotal accepted samples: "
        f"{total_samples}"
    )

    print(
        "\nCommands:"
    )

    print(
        "  Type exact Sinhala text -> accept label"
    )

    print(
        "  /bad  -> reject this crop"
    )

    print(
        "  /skip -> skip for now"
    )

    print(
        "  /quit -> save and stop"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Type exactly what is visible in the image."
    )

    print(
        "Do not correct or rewrite the news sentence."
    )

    # ========================================================
    # REVIEW LOOP
    # ========================================================

    for index in range(
        total_samples
    ):

        review_status = str(
            df.at[
                index,
                "review_status"
            ]
        ).strip()

        # Resume support
        if review_status in [
            "accepted",
            "rejected"
        ]:

            continue

        sample_id = str(
            df.at[
                index,
                "sample_id"
            ]
        )

        source_id = str(
            df.at[
                index,
                "source_id"
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

        print(
            "\n" + "=" * 72
        )

        print(
            f"Sample "
            f"{index + 1}/"
            f"{total_samples}"
        )

        print(
            f"Sample ID : "
            f"{sample_id}"
        )

        print(
            f"Source    : "
            f"{source_id}"
        )

        if not image_path.exists():

            print(
                "Image missing - automatically rejected."
            )

            df.at[
                index,
                "keep"
            ] = 0

            df.at[
                index,
                "review_status"
            ] = "rejected"

            save_dataframe(
                df
            )

            continue

        image = cv2.imread(
            str(
                image_path
            ),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                "Unreadable image - automatically rejected."
            )

            df.at[
                index,
                "keep"
            ] = 0

            df.at[
                index,
                "review_status"
            ] = "rejected"

            save_dataframe(
                df
            )

            continue

        display = prepare_display(
            image
        )

        window_name = (
            f"OCR Label - "
            f"{sample_id}"
        )

        cv2.imshow(
            window_name,
            display
        )

        cv2.waitKey(
            100
        )

        print(
            "\nLook at the image window."
        )

        user_input = input(
            "Ground truth > "
        ).strip()

        cv2.destroyWindow(
            window_name
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if user_input.lower() == "/quit":

            save_dataframe(
                df
            )

            cv2.destroyAllWindows()

            print(
                "\nProgress saved."
            )

            print(
                "You can run the script again "
                "to continue."
            )

            break

        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        if user_input.lower() == "/skip":

            df.at[
                index,
                "review_status"
            ] = "unreviewed"

            save_dataframe(
                df
            )

            print(
                "Skipped."
            )

            continue

        # ----------------------------------------------------
        # BAD CROP
        # ----------------------------------------------------

        if user_input.lower() == "/bad":

            df.at[
                index,
                "ground_truth"
            ] = ""

            df.at[
                index,
                "keep"
            ] = 0

            df.at[
                index,
                "review_status"
            ] = "rejected"

            save_dataframe(
                df
            )

            print(
                "Rejected."
            )

            continue

        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not user_input:

            print(
                "Empty label - left unreviewed."
            )

            save_dataframe(
                df
            )

            continue

        # ----------------------------------------------------
        # ACCEPT
        # ----------------------------------------------------

        df.at[
            index,
            "ground_truth"
        ] = user_input

        df.at[
            index,
            "keep"
        ] = 1

        df.at[
            index,
            "review_status"
        ] = "accepted"

        save_dataframe(
            df
        )

        print(
            "Label saved."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    cv2.destroyAllWindows()

    accepted_count = int(
        (
            df[
                "review_status"
            ]
            == "accepted"
        ).sum()
    )

    rejected_count = int(
        (
            df[
                "review_status"
            ]
            == "rejected"
        ).sum()
    )

    unreviewed_count = int(
        (
            df[
                "review_status"
            ]
            == "unreviewed"
        ).sum()
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "LABEL REVIEW STATUS"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTotal samples : "
        f"{total_samples}"
    )

    print(
        f"Accepted      : "
        f"{accepted_count}"
    )

    print(
        f"Rejected      : "
        f"{rejected_count}"
    )

    print(
        f"Unreviewed    : "
        f"{unreviewed_count}"
    )

    print(
        "\nSaved CSV:"
    )

    print(
        REVIEWED_CSV
    )
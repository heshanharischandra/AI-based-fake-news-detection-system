from pathlib import Path
import shutil

import cv2
import numpy as np
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

CANDIDATE_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "auto_line_candidates_012_031.csv"
)

REGION_DIR = (
    OCR_ROOT
    / "real_training_regions"
)

FILTER_ROOT = (
    OCR_ROOT
    / "real_training_auto_filtered"
)

ACCEPTED_DIR = (
    FILTER_ROOT
    / "accepted"
)

REJECTED_DIR = (
    FILTER_ROOT
    / "rejected"
)

METADATA_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
)

ACCEPTED_CSV = (
    METADATA_DIR
    / "auto_line_candidates_accepted.csv"
)

REJECTED_CSV = (
    METADATA_DIR
    / "auto_line_candidates_rejected.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_ABSOLUTE_WIDTH = 60
MIN_ABSOLUTE_HEIGHT = 12

MIN_WIDTH_RATIO = 0.15

MIN_HEIGHT_RATIO = 0.045
MAX_HEIGHT_RATIO = 0.45

MIN_ASPECT_RATIO = 1.20

MIN_CONTRAST_STD = 10.0

MIN_EDGE_DENSITY = 0.006
MAX_EDGE_DENSITY = 0.50

MIN_QUALITY_SCORE = 3


# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean_output():

    FILTER_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    ACCEPTED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REJECTED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for folder in [
        ACCEPTED_DIR,
        REJECTED_DIR
    ]:

        for file_path in folder.glob(
            "*.png"
        ):

            file_path.unlink()

    if ACCEPTED_CSV.exists():
        ACCEPTED_CSV.unlink()

    if REJECTED_CSV.exists():
        REJECTED_CSV.unlink()


# ============================================================
# IMAGE METRICS
# ============================================================

def calculate_edge_density(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    density = (
        np.count_nonzero(edges)
        / edges.size
    )

    return float(
        density
    )


def calculate_contrast(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.std(gray)
    )


# ============================================================
# QUALITY CHECK
# ============================================================

def evaluate_candidate(
    crop,
    region
):

    crop_height, crop_width = (
        crop.shape[:2]
    )

    region_height, region_width = (
        region.shape[:2]
    )

    width_ratio = (
        crop_width
        / max(
            region_width,
            1
        )
    )

    height_ratio = (
        crop_height
        / max(
            region_height,
            1
        )
    )

    aspect_ratio = (
        crop_width
        / max(
            crop_height,
            1
        )
    )

    contrast_std = (
        calculate_contrast(
            crop
        )
    )

    edge_density = (
        calculate_edge_density(
            crop
        )
    )

    quality_score = 0

    reasons = []

    # --------------------------------------------------------
    # Absolute size
    # --------------------------------------------------------

    if crop_width >= MIN_ABSOLUTE_WIDTH:
        quality_score += 1
    else:
        reasons.append(
            "too_narrow"
        )

    if crop_height < MIN_ABSOLUTE_HEIGHT:

        reasons.append(
            "too_short"
        )

    # --------------------------------------------------------
    # Width relative to headline region
    # --------------------------------------------------------

    if width_ratio >= MIN_WIDTH_RATIO:
        quality_score += 1
    else:
        reasons.append(
            "small_width_ratio"
        )

    # --------------------------------------------------------
    # Height relative to headline region
    # --------------------------------------------------------

    if (
        MIN_HEIGHT_RATIO
        <= height_ratio
        <= MAX_HEIGHT_RATIO
    ):

        quality_score += 1

    else:

        reasons.append(
            "bad_height_ratio"
        )

    # --------------------------------------------------------
    # Text-line shape
    # --------------------------------------------------------

    if aspect_ratio >= MIN_ASPECT_RATIO:

        quality_score += 1

    else:

        reasons.append(
            "poor_aspect_ratio"
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    if contrast_std >= MIN_CONTRAST_STD:

        quality_score += 1

    else:

        reasons.append(
            "low_contrast"
        )

    # --------------------------------------------------------
    # Edge activity
    # --------------------------------------------------------

    if (
        MIN_EDGE_DENSITY
        <= edge_density
        <= MAX_EDGE_DENSITY
    ):

        quality_score += 1

    else:

        reasons.append(
            "bad_edge_density"
        )

    # --------------------------------------------------------
    # Hard rejection conditions
    # --------------------------------------------------------

    hard_reject = (

        crop_width
        < MIN_ABSOLUTE_WIDTH

        or crop_height
        < MIN_ABSOLUTE_HEIGHT

        or height_ratio
        > MAX_HEIGHT_RATIO

        or edge_density
        < MIN_EDGE_DENSITY
    )

    accepted = (

        not hard_reject

        and quality_score
        >= MIN_QUALITY_SCORE
    )

    metrics = {

        "crop_width":
            crop_width,

        "crop_height":
            crop_height,

        "width_ratio":
            round(
                width_ratio,
                4
            ),

        "height_ratio":
            round(
                height_ratio,
                4
            ),

        "aspect_ratio":
            round(
                aspect_ratio,
                4
            ),

        "contrast_std":
            round(
                contrast_std,
                4
            ),

        "edge_density":
            round(
                edge_density,
                4
            ),

        "quality_score":
            quality_score,

        "filter_reasons":
            ",".join(
                reasons
            )
    }

    return (
        accepted,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "AUTOMATIC REAL OCR LINE QUALITY FILTER"
    )

    print(
        "=" * 72
    )

    clean_output()

    if not CANDIDATE_CSV.exists():

        raise FileNotFoundError(
            f"Candidate CSV not found:\n"
            f"{CANDIDATE_CSV}"
        )

    df = pd.read_csv(
        CANDIDATE_CSV,
        encoding="utf-8-sig"
    )

    print(
        f"\nCandidate rows: "
        f"{len(df)}"
    )

    accepted_rows = []

    rejected_rows = []

    # ========================================================
    # PROCESS
    # ========================================================

    for index, row in df.iterrows():

        sample_id = str(
            row[
                "sample_id"
            ]
        )

        source_id = str(
            row[
                "source_id"
            ]
        )

        image_path = (
            OCR_ROOT
            / str(
                row[
                    "image_path"
                ]
            )
        )

        region_path = (
            REGION_DIR
            / f"{source_id}_headline.png"
        )

        print(
            f"\n{index + 1}/"
            f"{len(df)} "
            f"{sample_id}"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not image_path.exists():

            print(
                "REJECTED - crop missing"
            )

            new_row = row.to_dict()

            new_row[
                "filter_status"
            ] = "rejected"

            new_row[
                "filter_reasons"
            ] = "crop_missing"

            rejected_rows.append(
                new_row
            )

            continue

        if not region_path.exists():

            print(
                "REJECTED - source region missing"
            )

            new_row = row.to_dict()

            new_row[
                "filter_status"
            ] = "rejected"

            new_row[
                "filter_reasons"
            ] = "region_missing"

            rejected_rows.append(
                new_row
            )

            continue

        crop = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        region = cv2.imread(
            str(region_path),
            cv2.IMREAD_COLOR
        )

        if crop is None or region is None:

            print(
                "REJECTED - unreadable image"
            )

            new_row = row.to_dict()

            new_row[
                "filter_status"
            ] = "rejected"

            new_row[
                "filter_reasons"
            ] = "unreadable"

            rejected_rows.append(
                new_row
            )

            continue

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        (
            accepted,
            metrics
        ) = evaluate_candidate(
            crop,
            region
        )

        new_row = (
            row.to_dict()
        )

        for key, value in (
            metrics.items()
        ):

            new_row[
                key
            ] = value

        filename = (
            image_path.name
        )

        # ----------------------------------------------------
        # Copy accepted/rejected
        # ----------------------------------------------------

        if accepted:

            new_row[
                "filter_status"
            ] = "accepted"

            destination = (
                ACCEPTED_DIR
                / filename
            )

            shutil.copy2(
                image_path,
                destination
            )

            accepted_rows.append(
                new_row
            )

            print(
                "ACCEPTED"
                f" | score="
                f"{metrics['quality_score']}"
                f" | edge="
                f"{metrics['edge_density']}"
            )

        else:

            new_row[
                "filter_status"
            ] = "rejected"

            destination = (
                REJECTED_DIR
                / filename
            )

            shutil.copy2(
                image_path,
                destination
            )

            rejected_rows.append(
                new_row
            )

            print(
                "REJECTED"
                f" | score="
                f"{metrics['quality_score']}"
                f" | "
                f"{metrics['filter_reasons']}"
            )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    accepted_df = pd.DataFrame(
        accepted_rows
    )

    rejected_df = pd.DataFrame(
        rejected_rows
    )

    accepted_df.to_csv(
        ACCEPTED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    rejected_df.to_csv(
        REJECTED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SOURCE COVERAGE
    # ========================================================

    if len(
        accepted_df
    ) > 0:

        source_counts = (
            accepted_df[
                "source_id"
            ]
            .value_counts()
            .sort_index()
        )

    else:

        source_counts = (
            pd.Series(
                dtype=int
            )
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "AUTO QUALITY FILTER COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTotal candidates : "
        f"{len(df)}"
    )

    print(
        f"Accepted         : "
        f"{len(accepted_df)}"
    )

    print(
        f"Rejected         : "
        f"{len(rejected_df)}"
    )

    print(
        f"Accepted sources : "
        f"{accepted_df['source_id'].nunique() if len(accepted_df) else 0}"
    )

    if len(
        source_counts
    ) > 0:

        print(
            "\nAccepted lines per source:"
        )

        print(
            source_counts.to_string()
        )

    print(
        "\nAccepted folder:"
    )

    print(
        ACCEPTED_DIR
    )

    print(
        "\nRejected folder:"
    )

    print(
        REJECTED_DIR
    )

    print(
        "\nAccepted CSV:"
    )

    print(
        ACCEPTED_CSV
    )

    print(
        "\nRejected CSV:"
    )

    print(
        REJECTED_CSV
    )

    print(
        "\nExisting clean training dataset "
        "was NOT changed."
    )
from pathlib import Path
import csv

import cv2

import auto_extract_new_real_text_lines as line_extractor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

LABELS_CSV = (
    DATASET_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ocr_dataset_audit"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "real_ocr_annotation_audit.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAFE INTEGER CONVERSION
#
# Some metadata rows may contain blank coordinate fields.
# Audit must report them instead of crashing.
# ============================================================

def safe_int(value):

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return int(value)

    except ValueError:

        try:
            return int(float(value))

        except ValueError:
            return None


# ============================================================
# LOAD CSV
# ============================================================

def load_records():

    records = []

    with open(
        LABELS_CSV,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            records.append(
                row
            )

    return records


# ============================================================
# IMAGE PATH
# ============================================================

def get_image_path(record):

    relative_path = str(
        record.get(
            "image_path",
            ""
        )
    ).strip()

    if not relative_path:

        return None

    return (
        DATASET_ROOT
        / Path(relative_path)
    )


# ============================================================
# CHECK COORDINATES
# ============================================================

def get_coordinates(record):

    left = safe_int(
        record.get(
            "left"
        )
    )

    top = safe_int(
        record.get(
            "top"
        )
    )

    right = safe_int(
        record.get(
            "right"
        )
    )

    bottom = safe_int(
        record.get(
            "bottom"
        )
    )

    if (
        left is None
        or top is None
        or right is None
        or bottom is None
    ):

        return None

    if (
        right <= left
        or bottom <= top
    ):

        return None

    return (
        left,
        top,
        right,
        bottom
    )


# ============================================================
# DETECT NUMBER OF VISUAL TEXT BANDS
# ============================================================

def detect_visual_lines(image_path):

    if image_path is None:

        return {
            "visual_line_count": 0,
            "status": "missing_image_path"
        }

    if not image_path.exists():

        return {
            "visual_line_count": 0,
            "status": "missing_image"
        }

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return {
            "visual_line_count": 0,
            "status": "image_read_failed"
        }

    try:

        mask = (
            line_extractor
            .create_text_mask(
                image
            )
        )

        bands = (
            line_extractor
            .find_line_bands(
                mask
            )
        )

        candidates = (
            line_extractor
            .filter_line_bands(
                mask,
                bands
            )
        )

        candidates = (
            line_extractor
            .merge_overlapping_candidates(
                candidates
            )
        )

        count = len(
            candidates
        )

        return {
            "visual_line_count":
                count,

            "status":
                "ok"
        }

    except Exception as error:

        return {
            "visual_line_count":
                0,

            "status":
                (
                    "analysis_error: "
                    + str(error)
                )
        }


# ============================================================
# RECTANGLE IOU
# ============================================================

def calculate_iou(
    record_a,
    record_b
):

    coords_a = (
        get_coordinates(
            record_a
        )
    )

    coords_b = (
        get_coordinates(
            record_b
        )
    )

    if (
        coords_a is None
        or coords_b is None
    ):

        return None

    (
        ax1,
        ay1,
        ax2,
        ay2
    ) = coords_a

    (
        bx1,
        by1,
        bx2,
        by2
    ) = coords_b

    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )

    intersection_width = max(
        0,
        intersection_x2
        - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2
        - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        * (by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        / float(
            union_area
        )
    )


# ============================================================
# FIND DUPLICATE / OVERLAPPING SAMPLES
# ============================================================

def find_duplicates(records):

    duplicate_map = {
        record.get(
            "sample_id",
            ""
        ): []
        for record in records
    }

    for i in range(
        len(records)
    ):

        record_a = records[i]

        for j in range(
            i + 1,
            len(records)
        ):

            record_b = records[j]

            # Only compare crops belonging to the same
            # original news source.
            if (
                record_a.get(
                    "source_id"
                )
                !=
                record_b.get(
                    "source_id"
                )
            ):

                continue

            iou = calculate_iou(
                record_a,
                record_b
            )

            # Missing coordinates:
            # cannot safely perform overlap comparison.
            if iou is None:
                continue

            text_a = str(
                record_a.get(
                    "ground_truth",
                    ""
                )
            ).strip()

            text_b = str(
                record_b.get(
                    "ground_truth",
                    ""
                )
            ).strip()

            same_text = (
                text_a
                == text_b
                and text_a != ""
            )

            duplicate_candidate = False

            reason = None

            # Same text + substantial overlap
            if (
                same_text
                and iou >= 0.50
            ):

                duplicate_candidate = True

                reason = (
                    "same_text_high_overlap"
                )

            # Extremely high overlap regardless of text
            elif iou >= 0.80:

                duplicate_candidate = True

                reason = (
                    "very_high_overlap"
                )

            if duplicate_candidate:

                sample_a = str(
                    record_a.get(
                        "sample_id",
                        ""
                    )
                )

                sample_b = str(
                    record_b.get(
                        "sample_id",
                        ""
                    )
                )

                duplicate_map[
                    sample_a
                ].append(
                    {
                        "other_sample":
                            sample_b,

                        "iou":
                            round(
                                iou,
                                4
                            ),

                        "reason":
                            reason
                    }
                )

                duplicate_map[
                    sample_b
                ].append(
                    {
                        "other_sample":
                            sample_a,

                        "iou":
                            round(
                                iou,
                                4
                            ),

                        "reason":
                            reason
                    }
                )

    return duplicate_map


# ============================================================
# AUDIT
# ============================================================

def run_audit():

    print("=" * 72)
    print("REAL SINHALA OCR DATASET AUDIT")
    print("=" * 72)

    print()
    print(
        "Labels:"
    )

    print(
        LABELS_CSV
    )

    records = load_records()

    print()
    print(
        "Total metadata records:",
        len(records)
    )

    print()
    print(
        "Checking duplicate / overlapping annotations..."
    )

    duplicate_map = (
        find_duplicates(
            records
        )
    )

    audit_rows = []

    multi_line_count = 0

    duplicate_count = 0

    missing_image_count = 0

    missing_coordinate_count = 0

    invalid_coordinate_count = 0

    analysis_error_count = 0

    # ========================================================
    # ANALYSE EVERY SAMPLE
    # ========================================================

    for index, record in enumerate(
        records,
        start=1
    ):

        sample_id = str(
            record.get(
                "sample_id",
                ""
            )
        ).strip()

        source_id = str(
            record.get(
                "source_id",
                ""
            )
        ).strip()

        image_path = (
            get_image_path(
                record
            )
        )

        # ----------------------------------------------------
        # COORDINATE STATUS
        # ----------------------------------------------------

        raw_coordinate_values = [
            record.get(
                "left"
            ),
            record.get(
                "top"
            ),
            record.get(
                "right"
            ),
            record.get(
                "bottom"
            ),
        ]

        has_blank_coordinate = any(

            value is None
            or str(
                value
            ).strip() == ""

            for value
            in raw_coordinate_values
        )

        coordinates = (
            get_coordinates(
                record
            )
        )

        if has_blank_coordinate:

            coordinate_status = (
                "missing_coordinates"
            )

            missing_coordinate_count += 1

        elif coordinates is None:

            coordinate_status = (
                "invalid_coordinates"
            )

            invalid_coordinate_count += 1

        else:

            coordinate_status = (
                "ok"
            )

        # ----------------------------------------------------
        # VISUAL LINE ANALYSIS
        # ----------------------------------------------------

        visual_info = (
            detect_visual_lines(
                image_path
            )
        )

        visual_line_count = (
            visual_info[
                "visual_line_count"
            ]
        )

        analysis_status = (
            visual_info[
                "status"
            ]
        )

        if analysis_status in {
            "missing_image",
            "missing_image_path",
            "image_read_failed"
        }:

            missing_image_count += 1

        elif analysis_status.startswith(
            "analysis_error"
        ):

            analysis_error_count += 1

        # ----------------------------------------------------
        # MULTI-LINE CANDIDATE
        #
        # IMPORTANT:
        # This is only an automatic flag for manual review.
        # It is NOT automatically considered wrong.
        # ----------------------------------------------------

        multi_line_candidate = (
            visual_line_count >= 2
        )

        if multi_line_candidate:

            multi_line_count += 1

        # ----------------------------------------------------
        # DUPLICATE CANDIDATE
        # ----------------------------------------------------

        duplicates = (
            duplicate_map.get(
                sample_id,
                []
            )
        )

        duplicate_candidate = (
            len(
                duplicates
            )
            > 0
        )

        if duplicate_candidate:

            duplicate_count += 1

        duplicate_details = " | ".join(

            (
                item[
                    "other_sample"
                ]
                + ":IOU="
                + str(
                    item[
                        "iou"
                    ]
                )
                + ":"
                + item[
                    "reason"
                ]
            )

            for item
            in duplicates
        )

        # ----------------------------------------------------
        # AUDIT FLAGS
        # ----------------------------------------------------

        audit_flags = []

        if multi_line_candidate:

            audit_flags.append(
                "MULTI_LINE"
            )

        if duplicate_candidate:

            audit_flags.append(
                "DUPLICATE_OVERLAP"
            )

        if coordinate_status != "ok":

            audit_flags.append(
                coordinate_status.upper()
            )

        if analysis_status != "ok":

            audit_flags.append(
                analysis_status.upper()
            )

        if audit_flags:

            overall_status = (
                "REVIEW"
            )

        else:

            overall_status = (
                "OK"
            )

        # ----------------------------------------------------
        # SAVE AUDIT ROW
        # ----------------------------------------------------

        audit_rows.append(
            {
                "sample_id":
                    sample_id,

                "source_id":
                    source_id,

                "line_number":
                    record.get(
                        "line_number",
                        ""
                    ),

                "image_path":
                    record.get(
                        "image_path",
                        ""
                    ),

                "ground_truth":
                    record.get(
                        "ground_truth",
                        ""
                    ),

                "left":
                    record.get(
                        "left",
                        ""
                    ),

                "top":
                    record.get(
                        "top",
                        ""
                    ),

                "right":
                    record.get(
                        "right",
                        ""
                    ),

                "bottom":
                    record.get(
                        "bottom",
                        ""
                    ),

                "crop_width":
                    record.get(
                        "crop_width",
                        ""
                    ),

                "crop_height":
                    record.get(
                        "crop_height",
                        ""
                    ),

                "coordinate_status":
                    coordinate_status,

                "visual_line_count":
                    visual_line_count,

                "multi_line_candidate":
                    multi_line_candidate,

                "duplicate_candidate":
                    duplicate_candidate,

                "duplicate_details":
                    duplicate_details,

                "analysis_status":
                    analysis_status,

                "overall_status":
                    overall_status,

                "audit_flags":
                    " | ".join(
                        audit_flags
                    )
            }
        )

        print(
            f"[{index:03d}/{len(records):03d}] "
            f"{sample_id} "
            f"| source={source_id} "
            f"| visual_lines={visual_line_count} "
            f"| multi={multi_line_candidate} "
            f"| duplicate={duplicate_candidate} "
            f"| coords={coordinate_status}"
        )

    # ========================================================
    # SAVE CSV REPORT
    # ========================================================

    fieldnames = [
        "sample_id",
        "source_id",
        "line_number",
        "image_path",
        "ground_truth",
        "left",
        "top",
        "right",
        "bottom",
        "crop_width",
        "crop_height",
        "coordinate_status",
        "visual_line_count",
        "multi_line_candidate",
        "duplicate_candidate",
        "duplicate_details",
        "analysis_status",
        "overall_status",
        "audit_flags",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            audit_rows
        )

    # ========================================================
    # SUSPICIOUS SAMPLES
    # ========================================================

    suspicious_samples = [

        row
        for row in audit_rows

        if row[
            "overall_status"
        ] == "REVIEW"
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    print(
        "Total samples:",
        len(
            records
        )
    )

    print(
        "Multi-line candidates:",
        multi_line_count
    )

    print(
        "Duplicate/overlap candidates:",
        duplicate_count
    )

    print(
        "Missing-coordinate samples:",
        missing_coordinate_count
    )

    print(
        "Invalid-coordinate samples:",
        invalid_coordinate_count
    )

    print(
        "Missing/unreadable images:",
        missing_image_count
    )

    print(
        "Analysis errors:",
        analysis_error_count
    )

    print(
        "Total samples requiring review:",
        len(
            suspicious_samples
        )
    )

    print()
    print(
        "Samples requiring review:"
    )

    for row in suspicious_samples:

        print(
            " ",
            row[
                "sample_id"
            ],
            "| source:",
            row[
                "source_id"
            ],
            "| visual_lines:",
            row[
                "visual_line_count"
            ],
            "| duplicate:",
            row[
                "duplicate_candidate"
            ],
            "| coordinates:",
            row[
                "coordinate_status"
            ],
            "| flags:",
            row[
                "audit_flags"
            ]
        )

    print()
    print(
        "Audit CSV saved:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_audit()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )
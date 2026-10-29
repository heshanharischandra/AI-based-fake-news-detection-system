from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np

import auto_extract_new_real_text_lines as line_extractor

from image_ocr import find_line_boxes
from ocr_v3_inference import get_ocr_v2_engine


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_hybrid"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_hybrid_debug"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    return image


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    sinhala_count = sum(
        1
        for c in text
        if "\u0D80" <= c <= "\u0DFF"
    )

    english_count = sum(
        1
        for c in text
        if (
            "A" <= c <= "Z"
            or
            "a" <= c <= "z"
        )
    )

    if sinhala_count > 0 and english_count > 0:
        return "si+en"

    if sinhala_count > 0:
        return "si"

    if english_count > 0:
        return "en"

    return "unknown"


# ============================================================
# EXTENDED COLOUR MASK
#
# Keep safe V3 top boundary = 52%
# Extend bottom from 86% -> 98.5%
# ============================================================

def create_extended_text_masks(image):

    height, width = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    yellow_lower = np.array(
        [14, 80, 100],
        dtype=np.uint8
    )

    yellow_upper = np.array(
        [48, 255, 255],
        dtype=np.uint8
    )

    white_lower = np.array(
        [0, 0, 165],
        dtype=np.uint8
    )

    white_upper = np.array(
        [180, 95, 255],
        dtype=np.uint8
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    roi_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    top_ratio = 0.52
    bottom_ratio = 0.985

    x1 = int(width * 0.02)
    x2 = int(width * 0.98)

    y1 = int(height * top_ratio)
    y2 = int(height * bottom_ratio)

    roi_mask[
        y1:y2,
        x1:x2
    ] = 255

    yellow_mask = cv2.bitwise_and(
        yellow_mask,
        roi_mask
    )

    white_mask = cv2.bitwise_and(
        white_mask,
        roi_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    scan_info = {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "top_ratio": top_ratio,
        "bottom_ratio": bottom_ratio
    }

    return (
        yellow_mask,
        white_mask,
        scan_info
    )


# ============================================================
# REMOVE DUPLICATE BOXES
# ============================================================

def remove_duplicate_boxes(boxes):

    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda b: (
            b["y"],
            b["x"]
        )
    )

    cleaned = []

    for box in boxes:

        center_y = (
            box["y"]
            + box["h"] / 2.0
        )

        duplicate_index = None

        for index, existing in enumerate(cleaned):

            existing_center_y = (
                existing["y"]
                + existing["h"] / 2.0
            )

            if abs(
                center_y
                - existing_center_y
            ) <= 15:

                duplicate_index = index
                break

        if duplicate_index is None:

            cleaned.append(
                box
            )

        else:

            old = cleaned[
                duplicate_index
            ]

            old_area = (
                old["w"]
                * old["h"]
            )

            new_area = (
                box["w"]
                * box["h"]
            )

            if new_area > old_area:

                cleaned[
                    duplicate_index
                ] = box

    cleaned.sort(
        key=lambda b:
            b["y"]
    )

    return cleaned


# ============================================================
# TEXT STRUCTURE ANALYSIS
# ============================================================

def analyse_text_structure(
    box,
    yellow_mask,
    white_mask
):

    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    if box["color"] == "yellow":
        source_mask = yellow_mask
    else:
        source_mask = white_mask

    region = source_mask[
        y:y + h,
        x:x + w
    ]

    if region.size == 0:
        return None

    foreground = cv2.countNonZero(
        region
    )

    total = (
        region.shape[0]
        * region.shape[1]
    )

    if total <= 0:
        return None

    density = (
        foreground
        / float(total)
    )

    binary = (
        region > 0
    ).astype(
        np.uint8
    )

    (
        component_count,
        labels,
        stats,
        centroids
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    useful_areas = []

    for i in range(
        1,
        component_count
    ):

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= 3:
            useful_areas.append(
                area
            )

    components = len(
        useful_areas
    )

    if foreground > 0 and useful_areas:

        largest_ratio = (
            max(useful_areas)
            / float(foreground)
        )

    else:

        largest_ratio = 1.0

    active_columns = (
        np.any(
            binary > 0,
            axis=0
        )
    )

    active_column_ratio = (
        np.count_nonzero(
            active_columns
        )
        / float(
            max(
                1,
                region.shape[1]
            )
        )
    )

    return {
        "density":
            float(density),

        "components":
            int(components),

        "largest_component_ratio":
            float(largest_ratio),

        "active_column_ratio":
            float(active_column_ratio)
    }


# ============================================================
# FILTER NON-TEXT OBJECTS
# ============================================================

def filter_text_like_boxes(
    boxes,
    yellow_mask,
    white_mask
):

    accepted = []

    for box in boxes:

        info = analyse_text_structure(
            box,
            yellow_mask,
            white_mask
        )

        if info is None:
            continue

        density = info[
            "density"
        ]

        components = info[
            "components"
        ]

        largest_ratio = info[
            "largest_component_ratio"
        ]

        active_columns = info[
            "active_column_ratio"
        ]

        if density < 0.015:
            continue

        if density > 0.60:
            continue

        if active_columns < 0.20:
            continue

        if (
            largest_ratio > 0.82
            and density > 0.12
        ):
            continue

        # ----------------------------------------------------
        # WIDE LOW-COMPONENT TEXT RECOVERY
        #
        # Old rule:
        #   width >= 180 and components < 3 -> reject
        #
        # Problem:
        # Bold Sinhala headline letters can be joined by the
        # colour mask / dilation, leaving only 2 connected
        # components even though the region is clearly a wide
        # text line.
        #
        # New conservative rule:
        # Allow a 2-component wide region only when it has
        # strong text-like evidence across the horizontal span.
        # One-component wide regions are still rejected.
        # ----------------------------------------------------

        wide_low_component_recovery = False

        if (
            box["w"] >= 180
            and components < 3
        ):

            strong_wide_text = (
                components >= 2
                and
                density >= 0.08
                and
                density <= 0.55
                and
                active_columns >= 0.55
                and
                largest_ratio <= 0.72
                and
                box["h"] >= 18
                and
                box["h"] <= 120
            )

            if not strong_wide_text:
                continue

            wide_low_component_recovery = True

        new_box = box.copy()

        new_box[
            "text_density"
        ] = round(
            density,
            4
        )

        new_box[
            "components"
        ] = components

        new_box[
            "largest_component_ratio"
        ] = round(
            largest_ratio,
            4
        )

        new_box[
            "active_column_ratio"
        ] = round(
            active_columns,
            4
        )

        new_box[
            "wide_low_component_recovery"
        ] = bool(
            wide_low_component_recovery
        )

        accepted.append(
            new_box
        )

    return accepted


# ============================================================
# REMOVE THIN OBJECTS
# ============================================================

def filter_thin_boxes(boxes):

    if not boxes:
        return []

    heights = np.array(
        [
            b["h"]
            for b in boxes
        ],
        dtype=np.float32
    )

    median_height = float(
        np.median(
            heights
        )
    )

    minimum_height = max(
        16,
        int(
            round(
                median_height * 0.40
            )
        )
    )

    filtered = [
        b
        for b in boxes
        if b["h"] >= minimum_height
    ]

    if not filtered:
        return boxes.copy()

    return filtered


# ============================================================
# VERTICAL CLUSTERS
# ============================================================

def create_vertical_clusters(boxes):

    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda b:
            b["y"]
    )

    heights = np.array(
        [
            b["h"]
            for b in boxes
        ],
        dtype=np.float32
    )

    median_height = float(
        np.median(
            heights
        )
    )

    gap_limit = max(
        22,
        int(
            round(
                median_height * 0.45
            )
        )
    )

    clusters = []

    current = [
        boxes[0]
    ]

    for box in boxes[1:]:

        previous = current[-1]

        previous_bottom = (
            previous["y"]
            + previous["h"]
        )

        gap = (
            box["y"]
            - previous_bottom
        )

        if gap <= gap_limit:

            current.append(
                box
            )

        else:

            clusters.append(
                current
            )

            current = [
                box
            ]

    clusters.append(
        current
    )

    return clusters


# ============================================================
# SELECT HEADLINE CLUSTER
# ============================================================

def select_best_colour_cluster(boxes):

    boxes = filter_thin_boxes(
        boxes
    )

    if not boxes:
        return []

    clusters = create_vertical_clusters(
        boxes
    )

    def score(cluster):

        line_count = len(
            cluster
        )

        total_width = sum(
            b["w"]
            for b in cluster
        )

        total_area = sum(
            b["w"] * b["h"]
            for b in cluster
        )

        average_y = sum(
            b["y"]
            for b in cluster
        ) / float(line_count)

        return (
            line_count,
            total_width,
            total_area,
            average_y
        )

    best = max(
        clusters,
        key=score
    )

    return sorted(
        best,
        key=lambda b:
            b["y"]
    )


# ============================================================
# RECOVER A MISSING PRECEDING HEADLINE LINE
#
# Why this exists:
# A yellow/white headline line can sometimes be merged with
# nearby photo/background colour pixels when the normal
# find_line_boxes() vertical dilation is used. The merged
# contour can become too tall and get rejected before it ever
# reaches the text-structure filter.
#
# Recovery strategy:
# - Do NOT change the stable normal detector.
# - Only after a valid headline cluster already exists, inspect
#   a narrow band immediately above the first selected line.
# - Use horizontal-only dilation, so nearby objects above/below
#   do not get vertically merged into the text line.
# - Accept only a wide, text-like candidate that sits close to
#   the existing headline cluster.
#
# This is deliberately conservative to reduce false positives.
# ============================================================

def recover_missing_preceding_colour_line(
    selected_boxes,
    yellow_mask,
    white_mask
):

    if len(selected_boxes) < 2:
        return None

    selected_boxes = sorted(
        selected_boxes,
        key=lambda b: b["y"]
    )

    image_height, image_width = (
        yellow_mask.shape[:2]
    )

    heights = np.array(
        [
            b["h"]
            for b in selected_boxes
        ],
        dtype=np.float32
    )

    median_height = float(
        np.median(
            heights
        )
    )

    centers = [
        (
            b["y"]
            + b["h"] / 2.0
        )
        for b in selected_boxes
    ]

    center_steps = [
        centers[index]
        - centers[index - 1]
        for index in range(
            1,
            len(centers)
        )
        if (
            centers[index]
            - centers[index - 1]
        ) > 0
    ]

    if center_steps:

        typical_step = float(
            np.median(
                np.array(
                    center_steps,
                    dtype=np.float32
                )
            )
        )

    else:

        typical_step = (
            median_height
            * 1.15
        )

    first_box = selected_boxes[0]

    first_top = int(
        first_box["y"]
    )

    search_depth = int(
        round(
            max(
                typical_step * 1.35,
                median_height * 1.50
            )
        )
    )

    search_y1 = max(
        0,
        first_top - search_depth
    )

    search_y2 = max(
        search_y1 + 1,
        first_top - 2
    )

    if search_y2 <= search_y1:
        return None

    minimum_width = max(
        120,
        int(
            round(
                image_width * 0.28
            )
        )
    )

    minimum_height = max(
        12,
        int(
            round(
                median_height * 0.22
            )
        )
    )

    maximum_height = min(
        max(
            110,
            int(
                round(
                    median_height * 1.45
                )
            )
        ),
        int(
            round(
                image_height * 0.16
            )
        )
    )

    maximum_gap = max(
        34,
        int(
            round(
                median_height * 0.70
            )
        )
    )

    horizontal_kernel_width = max(
        21,
        int(
            round(
                image_width * 0.042
            )
        )
    )

    if (
        horizontal_kernel_width
        % 2
        == 0
    ):
        horizontal_kernel_width += 1

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_kernel_width,
                1
            )
        )
    )

    candidates = []

    for color_name, source_mask in (
        (
            "yellow",
            yellow_mask
        ),
        (
            "white",
            white_mask
        )
    ):

        search_mask = np.zeros_like(
            source_mask
        )

        search_mask[
            search_y1:search_y2,
            :
        ] = source_mask[
            search_y1:search_y2,
            :
        ]

        joined = cv2.dilate(
            search_mask,
            horizontal_kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            joined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < minimum_width:
                continue

            if h < minimum_height:
                continue

            if h > maximum_height:
                continue

            candidate_bottom = (
                y + h
            )

            gap = (
                first_top
                - candidate_bottom
            )

            # The recovered line must be immediately above the
            # already-selected headline cluster.
            if gap < -4:
                continue

            if gap > maximum_gap:
                continue

            candidate = {
                "x":
                    int(x),

                "y":
                    int(y),

                "w":
                    int(w),

                "h":
                    int(h),

                "color":
                    color_name,

                "recovered":
                    True,

                "recovery_method":
                    (
                        "horizontal_only_"
                        "preceding_line_recovery"
                    )
            }

            info = analyse_text_structure(
                candidate,
                yellow_mask,
                white_mask
            )

            if info is None:
                continue

            density = info[
                "density"
            ]

            components = info[
                "components"
            ]

            largest_ratio = info[
                "largest_component_ratio"
            ]

            active_columns = info[
                "active_column_ratio"
            ]

            if density < 0.015:
                continue

            if density > 0.60:
                continue

            if active_columns < 0.20:
                continue

            if (
                largest_ratio > 0.82
                and density > 0.12
            ):
                continue

            if (
                w >= 180
                and components < 3
            ):
                continue

            # Avoid adding something that is effectively the
            # same line as an already-selected box.
            candidate_center_y = (
                y + h / 2.0
            )

            is_duplicate = False

            for existing in selected_boxes:

                existing_center_y = (
                    existing["y"]
                    + existing["h"] / 2.0
                )

                if abs(
                    candidate_center_y
                    - existing_center_y
                ) <= max(
                    14,
                    median_height * 0.25
                ):

                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            candidate[
                "text_density"
            ] = round(
                density,
                4
            )

            candidate[
                "components"
            ] = int(
                components
            )

            candidate[
                "largest_component_ratio"
            ] = round(
                largest_ratio,
                4
            )

            candidate[
                "active_column_ratio"
            ] = round(
                active_columns,
                4
            )

            candidate[
                "recovery_gap"
            ] = int(
                gap
            )

            # Prefer:
            # 1. a line close to the existing cluster,
            # 2. a wide line,
            # 3. more text-like components.
            candidate[
                "_recovery_score"
            ] = (
                -float(
                    abs(
                        gap
                    )
                ),
                float(
                    w
                ),
                float(
                    components
                )
            )

            candidates.append(
                candidate
            )

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item:
            item[
                "_recovery_score"
            ]
    )

    best = best.copy()

    best.pop(
        "_recovery_score",
        None
    )

    return best


# ============================================================
# COLOUR DETECTION
# ============================================================

def detect_colour_headline(image):

    (
        yellow_mask,
        white_mask,
        scan_info
    ) = create_extended_text_masks(
        image
    )

    # --------------------------------------------------------
    # NORMAL / STABLE DETECTION
    # --------------------------------------------------------

    yellow_boxes = find_line_boxes(
        yellow_mask,
        "yellow"
    )

    white_boxes = find_line_boxes(
        white_mask,
        "white"
    )

    raw_boxes = (
        yellow_boxes
        + white_boxes
    )

    raw_boxes = remove_duplicate_boxes(
        raw_boxes
    )

    text_like_boxes = filter_text_like_boxes(
        raw_boxes,
        yellow_mask,
        white_mask
    )

    selected_boxes = select_best_colour_cluster(
        text_like_boxes
    )

    # --------------------------------------------------------
    # PRECEDING-LINE RECOVERY GUARD
    #
    # Recovery is intentionally restricted to the case where
    # the normal detector found exactly 2 headline lines.
    #
    # Reason:
    # - unseen_001 genuinely had 3 lines but only 2 survived,
    #   so recovery is useful there.
    # - after the wide-line filter fix, unseen_002 already has
    #   all 3 genuine lines. Running recovery again can pull a
    #   bright photo/background region above the headline and
    #   create a false fourth line.
    #
    # This conservative guard avoids over-recovering once a
    # plausible 3-line headline cluster already exists.
    # --------------------------------------------------------

    recovered_box = None

    if len(selected_boxes) == 2:

        recovered_box = (
            recover_missing_preceding_colour_line(
                selected_boxes,
                yellow_mask,
                white_mask
            )
        )

    if recovered_box is not None:

        raw_boxes = (
            raw_boxes
            + [
                recovered_box
            ]
        )

        raw_boxes = remove_duplicate_boxes(
            raw_boxes
        )

        # Re-run the same structure filter so the recovered
        # candidate must satisfy the normal text-like rules.
        text_like_boxes = filter_text_like_boxes(
            raw_boxes,
            yellow_mask,
            white_mask
        )

        selected_boxes = (
            select_best_colour_cluster(
                text_like_boxes
            )
        )

    return (
        yellow_mask,
        white_mask,
        scan_info,
        raw_boxes,
        text_like_boxes,
        selected_boxes
    )


# ============================================================
# BUILD HEADLINE ROI
# ============================================================

def build_colour_roi(
    image,
    boxes
):

    image_height, image_width = (
        image.shape[:2]
    )

    min_x = min(
        b["x"]
        for b in boxes
    )

    max_x = max(
        b["x"] + b["w"]
        for b in boxes
    )

    min_y = min(
        b["y"]
        for b in boxes
    )

    max_y = max(
        b["y"] + b["h"]
        for b in boxes
    )

    heights = np.array(
        [
            b["h"]
            for b in boxes
        ],
        dtype=np.float32
    )

    median_height = float(
        np.median(
            heights
        )
    )

    x_margin = max(
        8,
        int(
            round(
                image_width * 0.015
            )
        )
    )

    y_margin = max(
        7,
        int(
            round(
                median_height * 0.18
            )
        )
    )

    x1 = max(
        0,
        min_x - x_margin
    )

    x2 = min(
        image_width,
        max_x + x_margin
    )

    y1 = max(
        0,
        min_y - y_margin
    )

    y2 = min(
        image_height,
        max_y + y_margin
    )

    roi = image[
        y1:y2,
        x1:x2
    ]

    info = {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "width": int(
            x2 - x1
        ),
        "height": int(
            y2 - y1
        )
    }

    return (
        roi,
        info
    )


# ============================================================
# TRAINING-STYLE SEGMENTATION
# ============================================================

def extract_training_style_lines(roi):

    mask = (
        line_extractor
        .create_text_mask(
            roi
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

    boxes = [
        line_extractor.expand_box(
            candidate,
            roi.shape
        )
        for candidate in candidates
    ]

    boxes = sorted(
        boxes,
        key=lambda b:
            b[1]
    )

    roi_height, roi_width = (
        roi.shape[:2]
    )

    crops = []

    for box in boxes:

        x1, y1, x2, y2 = box

        crop = roi[
            y1:y2 + 1,
            x1:x2 + 1
        ]

        if crop.size == 0:
            continue

        crop_height, crop_width = (
            crop.shape[:2]
        )

        if crop_height < 18:
            continue

        if crop_height > max(
            170,
            int(
                roi_height * 0.48
            )
        ):
            continue

        if crop_width < (
            roi_width * 0.20
        ):
            continue

        crops.append(
            {
                "line_number":
                    len(crops) + 1,

                "crop":
                    crop,

                "box":
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },

                "source":
                    "training_style_segmentation"
            }
        )

    return (
        mask,
        boxes,
        crops
    )


# ============================================================
# V4.2 ADAPTIVE COLOUR LINE CROPS
#
# IMPORTANT FIX:
#
# Do NOT give every line one common 652px width.
#
# Each detected line keeps its own geometry.
#
# The horizontal dilation used by find_line_boxes is
# compensated by trimming a small amount from both sides.
#
# White lines receive slightly more vertical padding because
# white thresholding can miss anti-aliased Sinhala marks.
# ============================================================

def create_adaptive_colour_crops(
    image,
    boxes
):

    image_height, image_width = (
        image.shape[:2]
    )

    boxes = sorted(
        boxes,
        key=lambda b:
            b["y"]
    )

    crops = []

    horizontal_trim = max(
        4,
        int(
            round(
                image_width * 0.01
            )
        )
    )

    for box in boxes:

        x = int(
            box["x"]
        )

        y = int(
            box["y"]
        )

        w = int(
            box["w"]
        )

        h = int(
            box["h"]
        )

        # ----------------------------------------------------
        # COMPENSATE HORIZONTAL DILATION
        # ----------------------------------------------------

        x1 = min(
            image_width - 1,
            x + horizontal_trim
        )

        x2 = max(
            x1 + 1,
            x + w - horizontal_trim
        )

        x2 = min(
            image_width,
            x2
        )

        # ----------------------------------------------------
        # YELLOW TEXT
        #
        # Usually detected with nearly correct full height.
        # Only a very small vertical trim is used.
        # ----------------------------------------------------

        if box["color"] == "yellow":

            vertical_trim = max(
                1,
                int(
                    round(
                        h * 0.015
                    )
                )
            )

            y1 = max(
                0,
                y + vertical_trim
            )

            y2 = min(
                image_height,
                y + h - vertical_trim
            )

        # ----------------------------------------------------
        # WHITE TEXT
        #
        # Add margin because thresholding can lose fine
        # Sinhala marks at top/bottom.
        # ----------------------------------------------------

        else:

            vertical_padding = max(
                6,
                int(
                    round(
                        h * 0.12
                    )
                )
            )

            y1 = max(
                0,
                y - vertical_padding
            )

            y2 = min(
                image_height,
                y + h + vertical_padding
            )

        if y2 <= y1:
            continue

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        crops.append(
            {
                "line_number":
                    len(crops) + 1,

                "crop":
                    crop,

                "box":
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },

                "detected_color":
                    box["color"],

                "source":
                    "adaptive_colour_line_crop"
            }
        )

    return crops


# ============================================================
# GENERIC FALLBACK
#
# Only used when colour detection cannot provide useful lines.
# ============================================================

def generic_window_fallback(image):

    image_height, image_width = (
        image.shape[:2]
    )

    windows = [
        (0.48, 0.80),
        (0.52, 0.86),
        (0.56, 0.92),
        (0.60, 0.96),
        (0.62, 0.985),
        (0.66, 0.99),
    ]

    attempts = []

    best_result = None
    best_score = -999999.0

    for number, (
        top_ratio,
        bottom_ratio
    ) in enumerate(
        windows,
        start=1
    ):

        y1 = int(
            image_height
            * top_ratio
        )

        y2 = int(
            image_height
            * bottom_ratio
        )

        roi = image[
            y1:y2,
            :
        ]

        if roi.size == 0:
            continue

        (
            mask,
            boxes,
            crops
        ) = extract_training_style_lines(
            roi
        )

        line_count = len(
            crops
        )

        score = (
            line_count * 100.0
        )

        for item in crops:

            crop_h, crop_w = (
                item[
                    "crop"
                ].shape[:2]
            )

            score += (
                crop_w
                / float(image_width)
            ) * 25.0

            if 28 <= crop_h <= 135:
                score += 20.0

            if crop_h > 170:
                score -= 40.0

        if line_count > 5:

            score -= (
                line_count - 5
            ) * 80.0

        attempts.append(
            {
                "window":
                    number,

                "top_ratio":
                    top_ratio,

                "bottom_ratio":
                    bottom_ratio,

                "line_count":
                    line_count,

                "score":
                    score
            }
        )

        if (
            line_count > 0
            and score > best_score
        ):

            best_score = score

            best_result = {
                "roi":
                    roi,

                "mask":
                    mask,

                "crops":
                    crops,

                "info":
                    {
                        "mode":
                            "generic_window_fallback",

                        "x1":
                            0,

                        "y1":
                            int(y1),

                        "x2":
                            int(image_width),

                        "y2":
                            int(y2),

                        "width":
                            int(image_width),

                        "height":
                            int(
                                y2 - y1
                            ),

                        "score":
                            float(score)
                    }
            }

    return (
        best_result,
        attempts
    )


# ============================================================
# DEBUG PREVIEW
# ============================================================

def create_preview(
    image,
    raw_boxes,
    text_boxes,
    selected_boxes
):

    preview = image.copy()

    # Raw = red
    for box in raw_boxes:

        cv2.rectangle(
            preview,
            (
                box["x"],
                box["y"]
            ),
            (
                box["x"] + box["w"],
                box["y"] + box["h"]
            ),
            (
                0,
                0,
                255
            ),
            2
        )

    # Text-like = blue
    for box in text_boxes:

        cv2.rectangle(
            preview,
            (
                box["x"],
                box["y"]
            ),
            (
                box["x"] + box["w"],
                box["y"] + box["h"]
            ),
            (
                255,
                0,
                0
            ),
            2
        )

    # Selected headline = green
    for number, box in enumerate(
        selected_boxes,
        start=1
    ):

        cv2.rectangle(
            preview,
            (
                box["x"],
                box["y"]
            ),
            (
                box["x"] + box["w"],
                box["y"] + box["h"]
            ),
            (
                0,
                255,
                0
            ),
            3
        )

        cv2.putText(
            preview,
            f"H{number}",
            (
                box["x"],
                max(
                    20,
                    box["y"] - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                0,
                255,
                0
            ),
            2,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

def run_ocr(image_path):

    image = load_image(
        image_path
    )

    image_height, image_width = (
        image.shape[:2]
    )

    engine = (
        get_ocr_v2_engine()
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    debug_dir = (
        DEBUG_DIR
        / f"run_{timestamp}"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("HYBRID AUTOMATIC IMAGE -> TEXT OCR V4.2")
    print("=" * 72)

    print(
        f"\nInput image size: "
        f"{image_width} x {image_height}"
    )

    print(
        "\nModel:"
    )

    print(
        engine.model_path
    )

    print(
        f"\nCheckpoint epoch: "
        f"{engine.checkpoint_epoch}"
    )

    if engine.validation_cer is not None:

        print(
            f"Validation CER: "
            f"{engine.validation_cer:.4f}"
        )

    if engine.validation_wer is not None:

        print(
            f"Validation WER: "
            f"{engine.validation_wer:.4f}"
        )

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print(
        "STEP 1: Automatic headline detection"
    )

    (
        yellow_mask,
        white_mask,
        scan_info,
        raw_boxes,
        text_boxes,
        selected_boxes
    ) = detect_colour_headline(
        image
    )

    print(
        "Colour scan area:"
    )

    print(
        scan_info
    )

    print(
        f"\nRaw candidates: "
        f"{len(raw_boxes)}"
    )

    for i, box in enumerate(
        raw_boxes,
        start=1
    ):

        print(
            f"  Raw {i}: {box}"
        )

    print(
        f"\nText-like candidates: "
        f"{len(text_boxes)}"
    )

    for i, box in enumerate(
        text_boxes,
        start=1
    ):

        print(
            f"  Text {i}: {box}"
        )

    print(
        f"\nSelected headline lines: "
        f"{len(selected_boxes)}"
    )

    for i, box in enumerate(
        selected_boxes,
        start=1
    ):

        print(
            f"  Headline {i}: {box}"
        )

    # ========================================================
    # DEBUG MASKS
    # ========================================================

    yellow_path = (
        debug_dir
        / "yellow_mask.png"
    )

    white_path = (
        debug_dir
        / "white_mask.png"
    )

    preview_path = (
        debug_dir
        / "headline_detection_preview.png"
    )

    cv2.imwrite(
        str(yellow_path),
        yellow_mask
    )

    cv2.imwrite(
        str(white_path),
        white_mask
    )

    preview = create_preview(
        image,
        raw_boxes,
        text_boxes,
        selected_boxes
    )

    cv2.imwrite(
        str(preview_path),
        preview
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print(
        "STEP 2: Creating OCR line crops"
    )

    line_crops = []

    extraction_method = None

    roi_info = None

    headline_roi = None

    segmentation_mask = None

    fallback_attempts = []

    if selected_boxes:

        (
            headline_roi,
            roi_info
        ) = build_colour_roi(
            image,
            selected_boxes
        )

        (
            segmentation_mask,
            segmentation_boxes,
            training_crops
        ) = extract_training_style_lines(
            headline_roi
        )

        print(
            f"Training-style lines found: "
            f"{len(training_crops)}"
        )

        # ----------------------------------------------------
        # BEST CASE
        # ----------------------------------------------------

        if (
            len(training_crops)
            == len(selected_boxes)
        ):

            line_crops = training_crops

            extraction_method = (
                "colour_roi_plus_training_segmentation"
            )

        # ----------------------------------------------------
        # IMPORTANT V4.2 FIX
        #
        # Instead of old common-width colour-boundary crops,
        # use adaptive per-line geometry.
        # ----------------------------------------------------

        else:

            print(
                "Training segmentation count mismatch."
            )

            print(
                "Using adaptive colour line crops."
            )

            line_crops = (
                create_adaptive_colour_crops(
                    image,
                    selected_boxes
                )
            )

            extraction_method = (
                "adaptive_colour_line_crop"
            )

    # ========================================================
    # COLOUR DETECTION FAILED
    # ========================================================

    if not line_crops:

        print()
        print(
            "Colour route failed."
        )

        print(
            "Trying generic fallback..."
        )

        (
            fallback,
            fallback_attempts
        ) = generic_window_fallback(
            image
        )

        if fallback is None:

            raise RuntimeError(
                "No usable OCR lines found."
            )

        headline_roi = (
            fallback[
                "roi"
            ]
        )

        segmentation_mask = (
            fallback[
                "mask"
            ]
        )

        line_crops = (
            fallback[
                "crops"
            ]
        )

        roi_info = (
            fallback[
                "info"
            ]
        )

        extraction_method = (
            "generic_window_fallback"
        )

    # ========================================================
    # SAVE ROI
    # ========================================================

    roi_path = (
        debug_dir
        / "selected_headline_roi.png"
    )

    if headline_roi is not None:

        cv2.imwrite(
            str(roi_path),
            headline_roi
        )

    mask_path = (
        debug_dir
        / "selected_segmentation_mask.png"
    )

    if segmentation_mask is not None:

        cv2.imwrite(
            str(mask_path),
            segmentation_mask
        )

    print()
    print(
        "Extraction method:"
    )

    print(
        extraction_method
    )

    print(
        f"OCR line crops: "
        f"{len(line_crops)}"
    )

    # ========================================================
    # STEP 3 — OCR
    # ========================================================

    print()
    print(
        "STEP 3: Running V3 CRNN OCR"
    )

    results = []

    for index, item in enumerate(
        line_crops,
        start=1
    ):

        crop = item[
            "crop"
        ]

        crop_path = (
            debug_dir
            / f"line_{index:02d}.png"
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        prediction = (
            engine.predict_line(
                crop_path
            )
        )

        text = (
            prediction[
                "text"
            ]
            .strip()
        )

        print()
        print("-" * 72)

        print(
            f"LINE {index}"
        )

        print("-" * 72)

        print(
            "Crop source:",
            item.get(
                "source",
                "unknown"
            )
        )

        print(
            "Crop shape:",
            crop.shape
        )

        print(
            "Prediction:"
        )

        print(
            text
        )

        results.append(
            {
                "line_number":
                    index,

                "text":
                    text,

                "crop_source":
                    item.get(
                        "source",
                        "unknown"
                    ),

                "crop_shape":
                    list(
                        crop.shape
                    ),

                "box":
                    item[
                        "box"
                    ],

                "crop_path":
                    str(
                        crop_path
                    )
            }
        )

    # ========================================================
    # COMBINE TEXT
    # ========================================================

    extracted_text = "\n".join(
        item[
            "text"
        ]
        for item in results
        if item[
            "text"
        ]
    )

    language = detect_language(
        extracted_text
    )

    # ========================================================
    # SAVE TXT
    # ========================================================

    txt_path = (
        RESULTS_DIR
        / f"extracted_text_{timestamp}.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            extracted_text
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        RESULTS_DIR
        / f"ocr_result_{timestamp}.json"
    )

    json_data = {

        "timestamp":
            datetime.now().isoformat(),

        "pipeline":
            "Hybrid Automatic Image OCR V4.2",

        "image_path":
            str(
                Path(
                    image_path
                ).resolve()
            ),

        "image_size":
            {
                "width":
                    image_width,

                "height":
                    image_height
            },

        "ocr_model":
            str(
                engine.model_path
            ),

        "checkpoint_epoch":
            engine.checkpoint_epoch,

        "validation_cer":
            engine.validation_cer,

        "validation_wer":
            engine.validation_wer,

        "colour_scan":
            scan_info,

        "raw_candidate_count":
            len(raw_boxes),

        "text_candidate_count":
            len(text_boxes),

        "selected_headline_count":
            len(selected_boxes),

        "extraction_method":
            extraction_method,

        "headline_roi":
            roi_info,

        "fallback_attempts":
            fallback_attempts,

        "ocr_line_count":
            len(results),

        "detected_language":
            language,

        "extracted_text":
            extracted_text,

        "txt_path":
            str(txt_path),

        "lines":
            results
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL EXTRACTED TEXT")
    print("=" * 72)

    print()

    if extracted_text:
        print(
            extracted_text
        )

    else:
        print(
            "No text extracted."
        )

    print(
        "\nDetected Language:",
        language
    )

    print(
        "Extraction Method:",
        extraction_method
    )

    print(
        "OCR Lines:",
        len(results)
    )

    print(
        "\nTXT saved:"
    )

    print(
        txt_path
    )

    print(
        "\nJSON saved:"
    )

    print(
        json_path
    )

    print(
        "\nDebug folder:"
    )

    print(
        debug_dir
    )

    print("=" * 72)

    return {
        "text":
            extracted_text,

        "language":
            language,

        "method":
            extraction_method,

        "txt_path":
            txt_path,

        "json_path":
            json_path
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        image_path = input(
            "\nEnter news image path: "
        ).strip().strip('"')

        run_ocr(
            image_path
        )

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )
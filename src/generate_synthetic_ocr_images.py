from pathlib import Path
from datetime import datetime

import csv
import html
import json
import random
import time

import pandas as pd

from playwright.sync_api import sync_playwright


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXT_POOL_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "metadata"
    / "synthetic_text_pool.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "synthetic_lines"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "metadata"
)

LABELS_CSV = (
    METADATA_DIR
    / "synthetic_labels.csv"
)

MANIFEST_JSON = (
    METADATA_DIR
    / "synthetic_manifest.json"
)


TOTAL_SAMPLES = 5000

RANDOM_SEED = 42


# ============================================================
# 2. VISUAL STYLES
# ============================================================

STYLES = [
    {
        "style_name": "yellow_dark",
        "text_color": "#FFD400",
        "background_color": "#171717",
        "font_size": 42,
        "font_weight": 600,
    },

    {
        "style_name": "white_dark",
        "text_color": "#FFFFFF",
        "background_color": "#181818",
        "font_size": 44,
        "font_weight": 600,
    },

    {
        "style_name": "yellow_black",
        "text_color": "#F8D500",
        "background_color": "#080808",
        "font_size": 46,
        "font_weight": 700,
    },

    {
        "style_name": "white_black",
        "text_color": "#F4F4F4",
        "background_color": "#050505",
        "font_size": 40,
        "font_weight": 500,
    },

    {
        "style_name": "black_white",
        "text_color": "#111111",
        "background_color": "#FFFFFF",
        "font_size": 42,
        "font_weight": 600,
    },

    {
        "style_name": "yellow_gray",
        "text_color": "#FFD817",
        "background_color": "#252525",
        "font_size": 38,
        "font_weight": 600,
    },

    {
        "style_name": "white_gray",
        "text_color": "#F7F7F7",
        "background_color": "#292929",
        "font_size": 46,
        "font_weight": 500,
    }
]


# ============================================================
# 3. CLEAR OLD SYNTHETIC FILES
# ============================================================

def clear_old_synthetic_files():

    deleted_count = 0

    for image_path in OUTPUT_DIR.glob(
        "synthetic_*.png"
    ):

        image_path.unlink()

        deleted_count += 1

    print(
        f"\nOld synthetic images removed: "
        f"{deleted_count}"
    )


# ============================================================
# 4. CREATE HTML
# ============================================================

def create_html(
    text,
    style
):

    safe_text = html.escape(
        text
    )

    return f"""
<!DOCTYPE html>

<html lang="si">

<head>

<meta charset="UTF-8">

<style>

html,
body {{

    margin: 0;

    padding: 0;

    background:
        {style["background_color"]};
}}

#ocr-line {{

    display: inline-block;

    padding:
        10px
        16px;

    font-family:
        "Nirmala UI",
        "Nirmala",
        sans-serif;

    font-size:
        {style["font_size"]}px;

    font-weight:
        {style["font_weight"]};

    line-height:
        1.35;

    color:
        {style["text_color"]};

    background:
        {style["background_color"]};

    white-space:
        nowrap;
}}

</style>

</head>

<body>

<div id="ocr-line">
{safe_text}
</div>

</body>

</html>
"""


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "FULL SYNTHETIC SINHALA OCR DATASET GENERATOR"
    )

    print(
        "=" * 72
    )


    # --------------------------------------------------------
    # Check source CSV
    # --------------------------------------------------------

    if not TEXT_POOL_CSV.exists():

        print(
            "\nERROR:"
        )

        print(
            "synthetic_text_pool.csv not found."
        )

        print(
            TEXT_POOL_CSV
        )

        raise SystemExit


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load text pool
    # --------------------------------------------------------

    df = pd.read_csv(
        TEXT_POOL_CSV
    )


    print(
        f"\nAvailable text samples: "
        f"{len(df)}"
    )


    if len(df) < TOTAL_SAMPLES:

        sample_count = len(df)

    else:

        sample_count = TOTAL_SAMPLES


    print(
        f"Images to generate: "
        f"{sample_count}"
    )


    # --------------------------------------------------------
    # Clean previous generated synthetic images
    # --------------------------------------------------------

    clear_old_synthetic_files()


    # --------------------------------------------------------
    # Reproducible dataset order
    # --------------------------------------------------------

    df_sample = df.sample(
        n=sample_count,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )


    random.seed(
        RANDOM_SEED
    )


    generated_rows = []


    start_time = time.time()


    # ========================================================
    # START MICROSOFT EDGE
    # ========================================================

    with sync_playwright() as p:

        browser = p.chromium.launch(
            channel="msedge",
            headless=True
        )


        page = browser.new_page(
            viewport={
                "width": 2200,
                "height": 300
            }
        )


        for index, row in df_sample.iterrows():

            sample_number = (
                index + 1
            )


            sample_id = (
                f"synthetic_"
                f"{sample_number:06d}"
            )


            text = str(
                row["text"]
            ).strip()


            # ------------------------------------------------
            # Select a visual style
            # ------------------------------------------------

            style = random.choice(
                STYLES
            )


            html_content = (
                create_html(
                    text,
                    style
                )
            )


            page.set_content(
                html_content,
                wait_until="load"
            )


            locator = page.locator(
                "#ocr-line"
            )


            image_filename = (
                f"{sample_id}.png"
            )


            image_path = (
                OUTPUT_DIR
                / image_filename
            )


            # ------------------------------------------------
            # Screenshot only text element
            # ------------------------------------------------

            locator.screenshot(
                path=str(
                    image_path
                )
            )


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            generated_rows.append({

                "sample_id":
                    sample_id,

                "image_path":
                    (
                        f"synthetic_lines/"
                        f"{image_filename}"
                    ),

                "ground_truth":
                    text,

                "language":
                    "si",

                "source_text_id":
                    row.get(
                        "text_id",
                        ""
                    ),

                "source":
                    row.get(
                        "source",
                        ""
                    ),

                "style_name":
                    style[
                        "style_name"
                    ],

                "font":
                    "Nirmala UI",

                "font_size":
                    style[
                        "font_size"
                    ],

                "font_weight":
                    style[
                        "font_weight"
                    ],

                "text_color":
                    style[
                        "text_color"
                    ],

                "background_color":
                    style[
                        "background_color"
                    ],

                "renderer":
                    (
                        "Microsoft Edge / Chromium"
                    )
            })


            # ------------------------------------------------
            # Progress output
            # ------------------------------------------------

            if (
                sample_number == 1
                or sample_number % 100 == 0
                or sample_number == sample_count
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"Generated: "
                    f"{sample_number}/"
                    f"{sample_count}"
                    f" | "
                    f"Elapsed: "
                    f"{elapsed:.1f}s"
                )


        browser.close()


    # ========================================================
    # SAVE LABELS CSV
    # ========================================================

    fieldnames = [

        "sample_id",

        "image_path",

        "ground_truth",

        "language",

        "source_text_id",

        "source",

        "style_name",

        "font",

        "font_size",

        "font_weight",

        "text_color",

        "background_color",

        "renderer"
    ]


    with open(
        LABELS_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            generated_rows
        )


    # ========================================================
    # SAVE JSON MANIFEST
    # ========================================================

    total_time = (
        time.time()
        - start_time
    )


    manifest = {

        "created_at":
            datetime.now().isoformat(),

        "dataset_type":
            "synthetic_ocr_full",

        "language":
            "si",

        "renderer":
            "Microsoft Edge / Chromium",

        "font":
            "Nirmala UI",

        "sample_count":
            len(
                generated_rows
            ),

        "random_seed":
            RANDOM_SEED,

        "generation_seconds":
            round(
                total_time,
                2
            ),

        "source_text_pool":
            str(
                TEXT_POOL_CSV.resolve()
            ),

        "labels_csv":
            str(
                LABELS_CSV.resolve()
            ),

        "samples":
            generated_rows
    }


    with open(
        MANIFEST_JSON,
        "w",
        encoding="utf-8"
    ) as json_file:

        json.dump(
            manifest,
            json_file,
            ensure_ascii=False,
            indent=4
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "FULL SYNTHETIC DATASET GENERATION COMPLETE"
    )

    print(
        "=" * 72
    )


    print(
        f"\nGenerated images: "
        f"{len(generated_rows)}"
    )


    print(
        f"Generation time: "
        f"{total_time:.1f} seconds"
    )


    print(
        "\nImages:"
    )

    print(
        OUTPUT_DIR
    )


    print(
        "\nLabels CSV:"
    )

    print(
        LABELS_CSV
    )


    print(
        "\nJSON manifest:"
    )

    print(
        MANIFEST_JSON
    )
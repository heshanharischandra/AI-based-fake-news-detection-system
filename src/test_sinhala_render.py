from pathlib import Path
from datetime import datetime
import json

from playwright.sync_api import sync_playwright


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEST TEXT
# ============================================================

TEST_TEXT = (
    "මහනුවර - යාපනය (A-9) මාර්ගය"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_IMAGE = (
    OUTPUT_DIR
    / "test_sinhala_render.png"
)

OUTPUT_JSON = (
    METADATA_DIR
    / "test_sinhala_render.json"
)


# ============================================================
# HTML
# ============================================================

HTML = f"""
<!DOCTYPE html>
<html lang="si">

<head>

<meta charset="UTF-8">

<style>

html, body {{
    margin: 0;
    padding: 0;
    background: #111111;
}}

#text-line {{

    display: inline-block;

    padding: 12px 18px;

    font-family:
        "Nirmala UI",
        "Nirmala",
        sans-serif;

    font-size: 46px;

    font-weight: 600;

    line-height: 1.3;

    color: #FFD400;

    background: #1A1A1A;

    white-space: nowrap;
}}

</style>

</head>


<body>

<div id="text-line">
{TEST_TEXT}
</div>

</body>

</html>
"""


# ============================================================
# SAVE JSON
# ============================================================

def save_json():

    json_data = {

        "timestamp":
            datetime.now().isoformat(),

        "sample_type":
            "synthetic_render_test",

        "language":
            "si",

        "ground_truth":
            TEST_TEXT,

        "font":
            "Nirmala UI",

        "font_size":
            46,

        "font_weight":
            600,

        "text_color":
            "#FFD400",

        "background_color":
            "#1A1A1A",

        "renderer":
            "Microsoft Edge / Chromium",

        "image_path":
            str(
                OUTPUT_IMAGE.resolve()
            )
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    return json_data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "SINHALA BROWSER RENDER TEST"
    )

    print(
        "=" * 60
    )

    print(
        "\nRendering Sinhala training sample..."
    )

    with sync_playwright() as p:

        print(
            "\nOpening Microsoft Edge..."
        )

        browser = p.chromium.launch(
            channel="msedge",
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1400,
                "height": 300
            }
        )

        page.set_content(
            HTML,
            wait_until="load"
        )

        text_element = page.locator(
            "#text-line"
        )

        text_element.screenshot(
            path=str(
                OUTPUT_IMAGE
            )
        )

        browser.close()

    json_data = save_json()

    print(
        "\nRender complete."
    )

    print(
        "\nGround Truth:"
    )

    print(
        TEST_TEXT
    )

    print(
        "\nImage saved:"
    )

    print(
        OUTPUT_IMAGE
    )

    print(
        "\nJSON saved:"
    )

    print(
        OUTPUT_JSON
    )

    print(
        "\n" + "=" * 60
    )
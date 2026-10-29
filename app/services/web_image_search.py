
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps
from google.cloud import vision


def is_web_url(value):
    if not value:
        return False

    parsed = urlparse(value)

    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
    )


def prepare_image_bytes(image_path):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")

        image.thumbnail((1600, 1600))

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=88,
            optimize=True
        )

        return buffer.getvalue()


def search_web_image(image_path, limit=5):
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if limit < 1:
        raise ValueError(
            "Limit must be at least 1."
        )

    print("[WEB SEARCH] Preparing uploaded image...")

    image_bytes = prepare_image_bytes(path)

    print("[WEB SEARCH] Connecting to Google Cloud Vision...")

    client = vision.ImageAnnotatorClient()

    response = client.web_detection(
        image=vision.Image(content=image_bytes)
    )

    if response.error.message:
        raise RuntimeError(
            response.error.message
        )

    web = response.web_detection

    print("[WEB SEARCH] Search completed.")

    pages = []
    seen = set()

    for page in web.pages_with_matching_images:
        url = page.url

        if not is_web_url(url) or url in seen:
            continue

        seen.add(url)

        if page.full_matching_images:
            match_type = "full_image_match"

        elif page.partial_matching_images:
            match_type = "partial_image_match"

        else:
            match_type = "page_with_matching_image"

        pages.append({
            "title": page.page_title or "Untitled page",
            "url": url,
            "match_type": match_type,
            "source_status": "not_independently_verified"
        })

        if len(pages) >= limit:
            break

    def image_urls(items):
        urls = []

        for item in items:
            if is_web_url(item.url):
                urls.append(item.url)

            if len(urls) >= limit:
                break

        return urls

    return {
        "status": "search_completed",
        "method": "google_cloud_vision_web_detection",
        "matched_pages": pages,
        "full_matching_images": image_urls(
            web.full_matching_images
        ),
        "partial_matching_images": image_urls(
            web.partial_matching_images
        ),
        "visually_similar_images": image_urls(
            web.visually_similar_images
        ),
        "best_guess_labels": [
            item.label
            for item in web.best_guess_labels
        ][:limit],
        "web_entities": [
            item.description
            for item in web.web_entities
            if item.description
        ][:limit],
        "note": (
            "Web matches are leads for human verification. "
            "They do not independently establish original "
            "date, event, location, or claim truth."
        )
    }


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Path of the image to search"
    )

    args = parser.parse_args()

    pprint(
        search_web_image(args.image)
    )

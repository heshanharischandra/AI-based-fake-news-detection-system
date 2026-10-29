import tempfile
import unittest
from pathlib import Path
from io import BytesIO
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from app.main import create_app, validate_image


class WebUiTests(unittest.TestCase):
    @staticmethod
    def image_bytes(format_name="PNG"):
        stream = BytesIO()
        Image.new("RGB", (24, 16), "white").save(stream, format=format_name)
        return stream.getvalue()

    def test_png_jpg_jpeg_are_accepted_case_insensitively(self):
        for filename, format_name, mime in (
            ("news.PNG", "PNG", "image/png"),
            ("news.jpg", "JPEG", "image/jpeg"),
            ("news.JPEG", "JPEG", "image/jpeg"),
            ("news.png", "PNG", "application/octet-stream"),
            ("news.jpeg", "JPEG", None),
        ):
            with self.subTest(filename=filename, mime=mime):
                info, preview = validate_image(self.image_bytes(format_name), filename, mime)
                self.assertEqual(info["format"], format_name)
                self.assertEqual(preview.size, (24, 16))

    def test_other_extensions_and_mime_types_are_rejected(self):
        for filename, mime in (("news.webp", "image/webp"), ("news.gif", "image/gif"),
                               ("news.pdf", "application/pdf"), ("news", "image/png"),
                               ("news.png.exe", "image/png"), ("news.png", "text/plain")):
            with self.subTest(filename=filename, mime=mime), self.assertRaises(HTTPException) as caught:
                validate_image(self.image_bytes(), filename, mime)
            self.assertEqual(caught.exception.status_code, 415)

    def test_disguised_images_and_mismatched_mime_are_rejected(self):
        cases = [("GIF", "news.png", "image/png"), ("BMP", "news.jpg", "image/jpeg"),
                 ("PNG", "news.jpeg", "image/jpeg"), ("JPEG", "news.jpg", "image/png")]
        for format_name, filename, mime in cases:
            with self.subTest(format_name=format_name), self.assertRaises(HTTPException) as caught:
                validate_image(self.image_bytes(format_name), filename, mime)
            self.assertEqual(caught.exception.status_code, 415)

    def test_empty_and_corrupt_images_are_rejected(self):
        for data in (b"", b"not an image", self.image_bytes()[:30]):
            with self.subTest(data=data), self.assertRaises(HTTPException) as caught:
                validate_image(data, "news.png", "image/png")
            self.assertEqual(caught.exception.status_code, 400)

    def test_byte_and_pixel_limits_are_enforced(self):
        for limit in ("MAX_UPLOAD", "MAX_PIXELS"):
            with patch("app.main." + limit, 10), self.assertRaises(HTTPException) as caught:
                validate_image(self.image_bytes(), "news.png", "image/png")
            self.assertEqual(caught.exception.status_code, 413)

    def test_app_serves_frontend_and_api_routes_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            application = create_app(upload_dir=folder, runner=lambda *args: None)
            routes = {route.path for route in application.routes}
            self.assertIn("/", routes)
            self.assertIn("/api/health", routes)
            self.assertIn("/api/analyses", routes)
            self.assertIn("/api/analyses/{job_id}", routes)
            self.assertTrue((Path(__file__).parents[1] / "app/web/index.html").is_file())
            self.assertTrue((Path(__file__).parents[1] / "app/web/app.js").is_file())
            application.state.jobs.close()


if __name__ == "__main__":
    unittest.main()

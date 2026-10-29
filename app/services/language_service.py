
import sys
import unicodedata


def normalize_text(text: str) -> str:
    # Preserve Sinhala vowel signs, virama and ZWJ used in conjunct rendering.
    return unicodedata.normalize("NFC", str(text or "")).replace("\ufeff", "").strip()


def detect_language(text: str) -> str:
    text = normalize_text(text)
    sinhala = any("\u0d80" <= c <= "\u0dff" and c.isalpha() for c in text)
    english = any(c.isascii() and c.isalpha() for c in text)
    return "si+en" if sinhala and english else "si" if sinhala else "en" if english else "unknown"


def configure_unicode_console():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

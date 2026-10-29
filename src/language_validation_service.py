from __future__ import annotations

import re
from typing import Dict, Any

try:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "si": "Sinhala",
}


# ============================================================
# UNICODE RANGES
# ============================================================

SINHALA_PATTERN = re.compile(
    r"[\u0D80-\u0DFF]"
)

TAMIL_PATTERN = re.compile(
    r"[\u0B80-\u0BFF]"
)

DEVANAGARI_PATTERN = re.compile(
    r"[\u0900-\u097F]"
)

ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06FF]"
)

CJK_PATTERN = re.compile(
    r"[\u4E00-\u9FFF]"
)

CYRILLIC_PATTERN = re.compile(
    r"[\u0400-\u04FF]"
)


# ============================================================
# HELPERS
# ============================================================

def count_matches(
    pattern: re.Pattern,
    text: str,
) -> int:

    return len(
        pattern.findall(
            text
        )
    )


def contains_unsupported_script(
    text: str,
) -> bool:

    unsupported_patterns = [
        TAMIL_PATTERN,
        DEVANAGARI_PATTERN,
        ARABIC_PATTERN,
        CJK_PATTERN,
        CYRILLIC_PATTERN,
    ]

    return any(
        pattern.search(
            text
        )
        for pattern in unsupported_patterns
    )


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validate_language(
    text: str,
) -> Dict[str, Any]:

    if text is None:

        return {
            "accepted": False,
            "language_code": None,
            "language_name": None,
            "message": (
                "No text was provided."
            ),
        }

    text = str(
        text
    ).strip()

    if not text:

        return {
            "accepted": False,
            "language_code": None,
            "language_name": None,
            "message": (
                "No readable text was found."
            ),
        }

    # ========================================================
    # 1. TAMIL CHECK
    # ========================================================

    if TAMIL_PATTERN.search(
        text
    ):

        return {
            "accepted": False,
            "language_code": "ta",
            "language_name": "Tamil",
            "message": (
                "Unsupported language. "
                "This system only supports "
                "English and Sinhala."
            ),
        }

    # ========================================================
    # 2. OTHER UNSUPPORTED SCRIPT CHECK
    # ========================================================

    if contains_unsupported_script(
        text
    ):

        return {
            "accepted": False,
            "language_code": "unsupported",
            "language_name": "Unsupported Language",
            "message": (
                "Unsupported language. "
                "This system only supports "
                "English and Sinhala."
            ),
        }

    # ========================================================
    # 3. SINHALA CHECK
    #
    # If Sinhala Unicode characters are present, treat the
    # content as Sinhala. English words/names may still appear
    # inside normal Sinhala news content.
    # ========================================================

    sinhala_count = count_matches(
        SINHALA_PATTERN,
        text,
    )

    if sinhala_count > 0:

        return {
            "accepted": True,
            "language_code": "si",
            "language_name": "Sinhala",
            "message": (
                "Supported language detected: Sinhala."
            ),
        }

    # ========================================================
    # 4. LATIN-SCRIPT LANGUAGE DETECTION
    #
    # Latin text could be English, French, Spanish, etc.
    # Use langdetect to reject non-English Latin languages.
    # ========================================================

    if detect is None:

        return {
            "accepted": False,
            "language_code": None,
            "language_name": None,
            "message": (
                "Language detector is not installed. "
                "Install langdetect before continuing."
            ),
        }

    try:

        detected_language = detect(
            text
        )

    except LangDetectException:

        return {
            "accepted": False,
            "language_code": None,
            "language_name": None,
            "message": (
                "Could not reliably detect the language. "
                "Please enter clear English or Sinhala news text."
            ),
        }

    # ========================================================
    # 5. ENGLISH ACCEPT
    # ========================================================

    if detected_language == "en":

        return {
            "accepted": True,
            "language_code": "en",
            "language_name": "English",
            "message": (
                "Supported language detected: English."
            ),
        }

    # ========================================================
    # 6. ALL OTHER LANGUAGES REJECT
    # ========================================================

    return {
        "accepted": False,
        "language_code": detected_language,
        "language_name": "Unsupported Language",
        "message": (
            "Unsupported language. "
            "This system only supports "
            "English and Sinhala."
        ),
    }


# ============================================================
# SIMPLE TERMINAL TEST
# ============================================================

def main():

    print("=" * 72)
    print("SINHALA / ENGLISH LANGUAGE VALIDATION")
    print("=" * 72)

    print()
    print(
        "Enter news text."
    )

    print(
        "Type EXIT to stop."
    )

    while True:

        print()

        user_text = input(
            "News Text: "
        ).strip()

        if user_text.upper() == "EXIT":

            break

        result = validate_language(
            user_text
        )

        print()
        print(
            "Accepted      :",
            result[
                "accepted"
            ],
        )

        print(
            "Language Code :",
            result[
                "language_code"
            ],
        )

        print(
            "Language      :",
            result[
                "language_name"
            ],
        )

        print(
            "Message       :",
            result[
                "message"
            ],
        )


if __name__ == "__main__":

    main()
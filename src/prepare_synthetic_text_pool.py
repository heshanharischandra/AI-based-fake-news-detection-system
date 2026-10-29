from pathlib import Path
import csv
import random
import re
import unicodedata

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "sinhala_dataset.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "metadata"
    / "synthetic_text_pool.csv"
)

MAX_SAMPLES = 5000

RANDOM_SEED = 42

MIN_CHARACTERS = 8
MAX_CHARACTERS = 75

MIN_WORDS = 2
MAX_WORDS = 12


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.replace(
        "\u200b",
        ""
    )

    text = text.replace(
        "\ufeff",
        ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def count_sinhala(text):

    return sum(
        1
        for ch in text
        if "\u0D80" <= ch <= "\u0DFF"
    )


def count_english(text):

    return sum(
        1
        for ch in text
        if (
            ch.isascii()
            and ch.isalpha()
        )
    )


def is_good_sinhala_text(text):

    if not text:
        return False

    character_count = len(
        text
    )

    word_count = len(
        text.split()
    )

    if character_count < MIN_CHARACTERS:
        return False

    if character_count > MAX_CHARACTERS:
        return False

    if word_count < MIN_WORDS:
        return False

    if word_count > MAX_WORDS:
        return False

    sinhala = count_sinhala(
        text
    )

    english = count_english(
        text
    )

    total_letters = (
        sinhala
        + english
    )

    if total_letters == 0:
        return False

    sinhala_ratio = (
        sinhala
        / total_letters
    )

    if sinhala_ratio < 0.70:
        return False

    return True


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = normalize_text(
        text
    )

    if not text:
        return []

    # Split on Sinhala/English sentence punctuation
    parts = re.split(
        r"(?<=[.!?])\s+|[\n\r]+",
        text
    )

    sentences = []

    for part in parts:

        part = normalize_text(
            part
        )

        if part:
            sentences.append(
                part
            )

    return sentences


# ============================================================
# SPLIT LONG SENTENCE INTO OCR-FRIENDLY CHUNKS
# ============================================================

def split_into_chunks(text):

    text = normalize_text(
        text
    )

    words = text.split()

    if not words:
        return []

    chunks = []

    current_words = []

    for word in words:

        candidate_words = (
            current_words
            + [word]
        )

        candidate = " ".join(
            candidate_words
        )

        # Keep line inside OCR-friendly width
        if (
            len(candidate)
            <= MAX_CHARACTERS
            and len(candidate_words)
            <= MAX_WORDS
        ):

            current_words.append(
                word
            )

        else:

            if current_words:

                chunk = " ".join(
                    current_words
                )

                chunks.append(
                    chunk
                )

            current_words = [
                word
            ]

    if current_words:

        chunks.append(
            " ".join(
                current_words
            )
        )

    return chunks


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "SYNTHETIC SINHALA OCR TEXT POOL PREPARATION"
    )

    print(
        "=" * 70
    )

    if not SOURCE_CSV.exists():

        print(
            "\nERROR:"
        )

        print(
            f"Dataset not found:\n{SOURCE_CSV}"
        )

        raise SystemExit

    # --------------------------------------------------------
    # Load original Sinhala dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        SOURCE_CSV
    )

    print(
        f"\nOriginal rows: {len(df)}"
    )

    required_columns = {
        "final_text",
        "language"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        print(
            "\nERROR:"
        )

        print(
            f"Missing columns: {missing_columns}"
        )

        raise SystemExit

    # --------------------------------------------------------
    # Sinhala rows only
    # --------------------------------------------------------

    df = df[
        df["language"]
        .astype(str)
        .str.lower()
        .eq("si")
    ].copy()

    print(
        f"Sinhala rows: {len(df)}"
    )

    candidates = []

    # --------------------------------------------------------
    # Process each article
    # --------------------------------------------------------

    for row_index, row in df.iterrows():

        final_text = normalize_text(
            row.get(
                "final_text",
                ""
            )
        )

        source = normalize_text(
            row.get(
                "source",
                ""
            )
        )

        binary_label = normalize_text(
            row.get(
                "binary_label",
                ""
            )
        )

        sentences = split_sentences(
            final_text
        )

        for sentence in sentences:

            chunks = split_into_chunks(
                sentence
            )

            for chunk in chunks:

                chunk = normalize_text(
                    chunk
                )

                if not is_good_sinhala_text(
                    chunk
                ):
                    continue

                candidates.append({
                    "text":
                        chunk,

                    "source":
                        source,

                    "original_row":
                        int(row_index),

                    "binary_label":
                        binary_label
                })

    print(
        f"Candidate lines before deduplication: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Remove duplicate text
    # --------------------------------------------------------

    unique_by_text = {}

    for item in candidates:

        text_key = item[
            "text"
        ]

        if text_key not in unique_by_text:

            unique_by_text[
                text_key
            ] = item

    candidates = list(
        unique_by_text.values()
    )

    print(
        f"Unique candidate lines: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Shuffle reproducibly
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        candidates
    )

    # --------------------------------------------------------
    # Limit dataset size
    # --------------------------------------------------------

    candidates = candidates[
        :MAX_SAMPLES
    ]

    # --------------------------------------------------------
    # Add IDs
    # --------------------------------------------------------

    output_rows = []

    for index, item in enumerate(
        candidates,
        start=1
    ):

        text_id = (
            f"text_{index:06d}"
        )

        output_rows.append({
            "text_id":
                text_id,

            "text":
                item[
                    "text"
                ],

            "language":
                "si",

            "source":
                item[
                    "source"
                ],

            "original_row":
                item[
                    "original_row"
                ],

            "original_binary_label":
                item[
                    "binary_label"
                ],

            "character_count":
                len(
                    item[
                        "text"
                    ]
                ),

            "word_count":
                len(
                    item[
                        "text"
                    ].split()
                )
        })

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "text_id",
        "text",
        "language",
        "source",
        "original_row",
        "original_binary_label",
        "character_count",
        "word_count"
    ]

    with open(
        OUTPUT_CSV,
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
            output_rows
        )

    # --------------------------------------------------------
    # Print sample
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TEXT POOL CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFinal text samples: "
        f"{len(output_rows)}"
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nFirst 10 samples:"
    )

    for item in output_rows[:10]:

        print(
            f"\n{item['text_id']}:"
        )

        print(
            item[
                "text"
            ]
        )
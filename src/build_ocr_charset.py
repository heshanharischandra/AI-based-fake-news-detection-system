from pathlib import Path
from datetime import datetime

import json
import unicodedata

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

SPLIT_DIR = (
    OCR_ROOT
    / "metadata"
    / "splits"
)

TRAIN_CSV = (
    SPLIT_DIR
    / "train.csv"
)

VALIDATION_CSV = (
    SPLIT_DIR
    / "validation.csv"
)

TEST_CSV = (
    SPLIT_DIR
    / "test.csv"
)

REAL_HOLDOUT_CSV = (
    SPLIT_DIR
    / "real_holdout.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


# ============================================================
# 2. NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace(
        "\ufeff",
        ""
    )

    text = text.replace(
        "\u200b",
        ""
    )

    text = unicodedata.normalize(
        "NFC",
        text
    )

    return text.strip()


# ============================================================
# 3. LOAD SPLIT
# ============================================================

def load_split(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig"
    )

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in df.columns
    ]

    if "ground_truth" not in df.columns:

        raise ValueError(
            f"'ground_truth' column missing in:\n{path}"
        )

    df["ground_truth"] = (
        df["ground_truth"]
        .apply(normalize_text)
    )

    return df


# ============================================================
# 4. EXTRACT CHARACTER SET
# ============================================================

def extract_characters(dataframe):

    characters = set()

    for text in dataframe[
        "ground_truth"
    ]:

        for character in text:

            characters.add(
                character
            )

    return characters


# ============================================================
# 5. CHARACTER TYPE
# ============================================================

def get_character_type(character):

    codepoint = ord(
        character
    )

    if character == " ":
        return "space"

    if (
        0x0D80
        <= codepoint
        <= 0x0DFF
    ):
        return "sinhala"

    if (
        character.isascii()
        and character.isalpha()
    ):
        return "english"

    if character.isdigit():
        return "digit"

    if character.isspace():
        return "whitespace"

    return "punctuation_or_symbol"


# ============================================================
# 6. CHECK UNSEEN CHARACTERS
# ============================================================

def find_unseen_characters(
    dataframe,
    training_characters
):

    split_characters = (
        extract_characters(
            dataframe
        )
    )

    unseen = (
        split_characters
        - training_characters
    )

    return sorted(
        unseen,
        key=lambda char:
            ord(char)
    )


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "SINHALA OCR CHARACTER VOCABULARY BUILDER"
    )

    print(
        "=" * 72
    )


    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_split(
        TRAIN_CSV
    )

    validation_df = load_split(
        VALIDATION_CSV
    )

    test_df = load_split(
        TEST_CSV
    )

    real_df = load_split(
        REAL_HOLDOUT_CSV
    )


    print(
        f"\nTrain samples      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_df)}"
    )

    print(
        f"Test samples       : "
        f"{len(test_df)}"
    )

    print(
        f"Real holdout       : "
        f"{len(real_df)}"
    )


    # --------------------------------------------------------
    # Build charset ONLY from training data
    # --------------------------------------------------------

    training_characters = (
        extract_characters(
            train_df
        )
    )


    # Sort by Unicode codepoint
    sorted_characters = sorted(
        training_characters,
        key=lambda char:
            ord(char)
    )


    # --------------------------------------------------------
    # CTC INDEXING
    #
    # Index 0 is reserved for CTC blank.
    # Real characters start from index 1.
    # --------------------------------------------------------

    char_to_index = {}

    index_to_char = {}


    for index, character in enumerate(
        sorted_characters,
        start=1
    ):

        char_to_index[
            character
        ] = index

        index_to_char[
            str(index)
        ] = character


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    sinhala_characters = [
        char
        for char in sorted_characters
        if get_character_type(char)
        == "sinhala"
    ]

    english_characters = [
        char
        for char in sorted_characters
        if get_character_type(char)
        == "english"
    ]

    digit_characters = [
        char
        for char in sorted_characters
        if get_character_type(char)
        == "digit"
    ]

    symbol_characters = [
        char
        for char in sorted_characters
        if get_character_type(char)
        == "punctuation_or_symbol"
    ]


    # --------------------------------------------------------
    # Maximum text lengths
    # --------------------------------------------------------

    train_max_length = max(
        len(text)
        for text in train_df[
            "ground_truth"
        ]
    )

    validation_max_length = max(
        len(text)
        for text in validation_df[
            "ground_truth"
        ]
    )

    test_max_length = max(
        len(text)
        for text in test_df[
            "ground_truth"
        ]
    )

    real_max_length = max(
        len(text)
        for text in real_df[
            "ground_truth"
        ]
    )


    # --------------------------------------------------------
    # Check unseen characters
    # --------------------------------------------------------

    validation_unseen = (
        find_unseen_characters(
            validation_df,
            training_characters
        )
    )

    test_unseen = (
        find_unseen_characters(
            test_df,
            training_characters
        )
    )

    real_unseen = (
        find_unseen_characters(
            real_df,
            training_characters
        )
    )


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    charset_data = {

        "created_at":
            datetime.now().isoformat(),

        "normalization":
            "NFC",

        "ctc_blank_index":
            0,

        "character_start_index":
            1,

        "character_count":
            len(
                sorted_characters
            ),

        "model_output_classes":
            len(
                sorted_characters
            ) + 1,

        "characters":
            sorted_characters,

        "char_to_index":
            char_to_index,

        "index_to_char":
            index_to_char,

        "statistics": {

            "sinhala_characters":
                len(
                    sinhala_characters
                ),

            "english_characters":
                len(
                    english_characters
                ),

            "digit_characters":
                len(
                    digit_characters
                ),

            "symbol_characters":
                len(
                    symbol_characters
                ),

            "train_max_text_length":
                train_max_length,

            "validation_max_text_length":
                validation_max_length,

            "test_max_text_length":
                test_max_length,

            "real_holdout_max_text_length":
                real_max_length
        },

        "unseen_characters": {

            "validation":
                validation_unseen,

            "test":
                test_unseen,

            "real_holdout":
                real_unseen
        }
    }


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        CHARSET_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            charset_data,
            file,
            ensure_ascii=False,
            indent=4
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "CHARSET CREATED"
    )

    print(
        "=" * 72
    )


    print(
        f"\nUnique OCR characters : "
        f"{len(sorted_characters)}"
    )

    print(
        f"CTC blank index       : 0"
    )

    print(
        f"Model output classes  : "
        f"{len(sorted_characters) + 1}"
    )


    print(
        f"\nSinhala characters    : "
        f"{len(sinhala_characters)}"
    )

    print(
        f"English characters    : "
        f"{len(english_characters)}"
    )

    print(
        f"Digits                : "
        f"{len(digit_characters)}"
    )

    print(
        f"Symbols/punctuation   : "
        f"{len(symbol_characters)}"
    )


    print(
        "\nMaximum text lengths:"
    )

    print(
        f"Train                 : "
        f"{train_max_length}"
    )

    print(
        f"Validation            : "
        f"{validation_max_length}"
    )

    print(
        f"Test                  : "
        f"{test_max_length}"
    )

    print(
        f"Real holdout          : "
        f"{real_max_length}"
    )


    print(
        "\nUnseen characters:"
    )

    print(
        f"Validation : "
        f"{validation_unseen}"
    )

    print(
        f"Test       : "
        f"{test_unseen}"
    )

    print(
        f"Real       : "
        f"{real_unseen}"
    )


    print(
        "\nCharset saved:"
    )

    print(
        CHARSET_JSON
    )


    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if (
        not validation_unseen
        and not test_unseen
        and not real_unseen
    ):

        print(
            "\nCHARSET COVERAGE: PASSED"
        )

        print(
            "All validation, test and real-holdout "
            "characters are represented in the "
            "training vocabulary."
        )

    else:

        print(
            "\nCHARSET COVERAGE: NEEDS REVIEW"
        )

        print(
            "One or more evaluation splits contain "
            "characters not present in training."
        )
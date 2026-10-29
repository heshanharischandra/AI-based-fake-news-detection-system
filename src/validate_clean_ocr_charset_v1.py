from pathlib import Path
import csv
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

CHARSET_PATH = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)

TRAIN_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "metadata"
    / "splits"
    / "clean_real_train_v1.csv"
)

VALIDATION_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "metadata"
    / "splits"
    / "clean_real_validation_v1.csv"
)


# ============================================================
# LOAD CHARSET
# ============================================================

def load_charset():

    if not CHARSET_PATH.exists():
        raise FileNotFoundError(
            f"Charset file not found:\n{CHARSET_PATH}"
        )

    with open(
        CHARSET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Support common charset JSON formats.
    if isinstance(data, list):

        characters = set(
            str(item)
            for item in data
        )

    elif isinstance(data, dict):

        # Example:
        # {"char_to_idx": {...}}
        if "char_to_idx" in data:

            characters = set(
                data["char_to_idx"].keys()
            )

        # Example:
        # {"characters": [...]}
        elif "characters" in data:

            characters = set(
                str(item)
                for item in data["characters"]
            )

        # Example direct mapping:
        # {"අ": 1, "ක": 2, ...}
        else:

            characters = set(
                str(key)
                for key in data.keys()
            )

    else:

        raise ValueError(
            "Unsupported charset.json format."
        )

    return characters


# ============================================================
# LOAD CSV
# ============================================================

def load_records(path):

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ============================================================
# FIND UNKNOWN CHARACTERS
# ============================================================

def analyse_records(
    records,
    charset
):

    unknown_characters = set()

    affected_samples = []

    total_characters = 0

    for record in records:

        sample_id = str(
            record.get(
                "sample_id",
                ""
            )
        ).strip()

        text = str(
            record.get(
                "ground_truth",
                ""
            )
        )

        sample_unknown = []

        for character in text:

            total_characters += 1

            # Spaces do not need a CTC class if the
            # existing model does not explicitly encode them.
            # But if space exists in charset, that is fine.
            if (
                character == " "
                and character not in charset
            ):
                continue

            if character not in charset:

                unknown_characters.add(
                    character
                )

                sample_unknown.append(
                    character
                )

        if sample_unknown:

            affected_samples.append(
                {
                    "sample_id":
                        sample_id,

                    "ground_truth":
                        text,

                    "unknown":
                        sorted(
                            set(
                                sample_unknown
                            )
                        )
                }
            )

    return {
        "total_characters":
            total_characters,

        "unknown_characters":
            sorted(
                unknown_characters
            ),

        "affected_samples":
            affected_samples,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("CLEAN OCR CHARSET VALIDATION V1")
    print("=" * 72)

    charset = load_charset()

    train_records = load_records(
        TRAIN_CSV
    )

    validation_records = load_records(
        VALIDATION_CSV
    )

    print()
    print(
        "Charset size:",
        len(charset)
    )

    print(
        "Train records:",
        len(train_records)
    )

    print(
        "Validation records:",
        len(validation_records)
    )

    train_result = analyse_records(
        train_records,
        charset
    )

    validation_result = analyse_records(
        validation_records,
        charset
    )

    all_unknown = sorted(
        set(
            train_result[
                "unknown_characters"
            ]
        )
        |
        set(
            validation_result[
                "unknown_characters"
            ]
        )
    )

    print()
    print("=" * 72)
    print("TRAIN CHECK")
    print("=" * 72)

    print(
        "Unknown characters:",
        train_result[
            "unknown_characters"
        ]
    )

    print(
        "Affected train samples:",
        len(
            train_result[
                "affected_samples"
            ]
        )
    )

    for item in train_result[
        "affected_samples"
    ]:

        print()
        print(
            item[
                "sample_id"
            ]
        )

        print(
            "GT:",
            item[
                "ground_truth"
            ]
        )

        print(
            "Unknown:",
            item[
                "unknown"
            ]
        )

    print()
    print("=" * 72)
    print("VALIDATION CHECK")
    print("=" * 72)

    print(
        "Unknown characters:",
        validation_result[
            "unknown_characters"
        ]
    )

    print(
        "Affected validation samples:",
        len(
            validation_result[
                "affected_samples"
            ]
        )
    )

    for item in validation_result[
        "affected_samples"
    ]:

        print()
        print(
            item[
                "sample_id"
            ]
        )

        print(
            "GT:",
            item[
                "ground_truth"
            ]
        )

        print(
            "Unknown:",
            item[
                "unknown"
            ]
        )

    print()
    print("=" * 72)

    if all_unknown:

        print(
            "CHARSET VALIDATION FAILED"
        )

        print(
            "Unknown characters found:"
        )

        for character in all_unknown:

            print(
                repr(character)
            )

        print()
        print(
            "Do NOT train yet."
        )

    else:

        print(
            "CHARSET VALIDATION PASSED"
        )

        print()
        print(
            "All cleaned train/validation labels "
            "are compatible with the current OCR charset."
        )

        print()
        print(
            "Dataset is ready for NEW clean-data augmentation."
        )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )
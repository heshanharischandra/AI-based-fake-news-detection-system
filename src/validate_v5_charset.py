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

V5_CSV = (
    OCR_ROOT
    / "real_training_expansion_v5"
    / "metadata"
    / "v5_verified_real_lines.csv"
)


# ============================================================
# LOAD CHARSET
# ============================================================

def load_charset():

    if not CHARSET_PATH.exists():

        raise FileNotFoundError(
            f"Charset file not found:\n"
            f"{CHARSET_PATH}"
        )

    with open(
        CHARSET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    # --------------------------------------------------------
    # SUPPORT COMMON CHARSET FORMATS
    # --------------------------------------------------------

    if isinstance(
        data,
        list,
    ):

        characters = set(
            str(item)
            for item in data
        )

    elif isinstance(
        data,
        dict,
    ):

        if "char_to_idx" in data:

            characters = set(
                data[
                    "char_to_idx"
                ].keys()
            )

        elif "characters" in data:

            characters = set(
                str(item)
                for item in data[
                    "characters"
                ]
            )

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

    if not path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n"
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# FIND UNKNOWN CHARACTERS
# ============================================================

def analyse_records(
    records,
    charset,
):

    unknown_characters = set()

    affected_samples = []

    total_characters = 0

    for record in records:

        sample_id = str(
            record.get(
                "sample_id",
                "",
            )
        ).strip()

        text = str(
            record.get(
                "text",
                "",
            )
        )

        sample_unknown = []

        for character in text:

            total_characters += 1

            # ------------------------------------------------
            # IGNORE SPACE ONLY IF CURRENT CHARSET
            # DOES NOT EXPLICITLY STORE IT
            # ------------------------------------------------

            if (
                character == " "
                and
                character not in charset
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

                    "text":
                        text,

                    "unknown":
                        sorted(
                            set(
                                sample_unknown
                            )
                        ),
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
    print("V5 OCR CHARSET VALIDATION")
    print("=" * 72)

    charset = load_charset()

    records = load_records(
        V5_CSV
    )

    print()
    print(
        "Charset size:",
        len(
            charset
        ),
    )

    print(
        "V5 verified records:",
        len(
            records
        ),
    )

    result = analyse_records(
        records,
        charset,
    )

    print()
    print("=" * 72)
    print("V5 CHECK")
    print("=" * 72)

    print(
        "Total characters:",
        result[
            "total_characters"
        ],
    )

    print(
        "Unknown characters:",
        result[
            "unknown_characters"
        ],
    )

    print(
        "Affected samples:",
        len(
            result[
                "affected_samples"
            ]
        ),
    )

    for item in result[
        "affected_samples"
    ]:

        print()
        print(
            "Sample:",
            item[
                "sample_id"
            ],
        )

        print(
            "Text:",
            item[
                "text"
            ],
        )

        print(
            "Unknown:",
            item[
                "unknown"
            ],
        )

    print()
    print("=" * 72)

    if result[
        "unknown_characters"
    ]:

        print(
            "V5 CHARSET VALIDATION FAILED"
        )

        print()

        print(
            "Unknown characters found:"
        )

        for character in result[
            "unknown_characters"
        ]:

            print(
                repr(
                    character
                )
            )

        print()

        print(
            "Do NOT start V5 training yet."
        )

    else:

        print(
            "V5 CHARSET VALIDATION PASSED"
        )

        print()

        print(
            "All 45 verified V5 labels are "
            "compatible with the current OCR charset."
        )

        print()

        print(
            "V5 real OCR data is ready for "
            "augmentation / training preparation."
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
        print("ERROR:")
        print(error)
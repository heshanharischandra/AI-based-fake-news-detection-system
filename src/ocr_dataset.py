from pathlib import Path
import json
import unicodedata

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


IMAGE_HEIGHT = 48
IMAGE_WIDTH = 1024


# ============================================================
# TEXT NORMALIZATION
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
# LOAD CHARSET
# ============================================================

def load_charset():

    if not CHARSET_JSON.exists():

        raise FileNotFoundError(
            f"charset.json not found:\n"
            f"{CHARSET_JSON}"
        )

    with open(
        CHARSET_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        charset_data = json.load(
            file
        )

    char_to_index = (
        charset_data[
            "char_to_index"
        ]
    )

    index_to_char = {
        int(index):
            character

        for index, character
        in charset_data[
            "index_to_char"
        ].items()
    }

    blank_index = (
        charset_data[
            "ctc_blank_index"
        ]
    )

    return (
        charset_data,
        char_to_index,
        index_to_char,
        blank_index
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    original_height, original_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Preserve aspect ratio
    # --------------------------------------------------------

    scale = (
        IMAGE_HEIGHT
        / original_height
    )

    resized_width = int(
        original_width
        * scale
    )

    # Prevent image becoming wider than model input
    resized_width = min(
        resized_width,
        IMAGE_WIDTH
    )

    resized_width = max(
        resized_width,
        1
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            IMAGE_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # White padded canvas
    # --------------------------------------------------------

    canvas = np.full(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ),
        255,
        dtype=np.uint8
    )

    canvas[
        :,
        :resized_width
    ] = resized

    # --------------------------------------------------------
    # Normalize 0-255 → 0-1
    # --------------------------------------------------------

    canvas = (
        canvas.astype(
            np.float32
        )
        / 255.0
    )

    # --------------------------------------------------------
    # Normalize approximately to -1 → +1
    # --------------------------------------------------------

    canvas = (
        canvas - 0.5
    ) / 0.5

    # Add channel dimension
    # [H, W] → [1, H, W]

    tensor = torch.from_numpy(
        canvas
    ).unsqueeze(
        0
    )

    return tensor


# ============================================================
# OCR DATASET
# ============================================================

class OCRDataset(Dataset):

    def __init__(
        self,
        csv_path
    ):

        self.csv_path = Path(
            csv_path
        )

        if not self.csv_path.exists():

            raise FileNotFoundError(
                f"Dataset CSV not found:\n"
                f"{self.csv_path}"
            )

        self.df = pd.read_csv(
            self.csv_path,
            encoding="utf-8-sig"
        )

        self.df.columns = [
            str(column)
            .replace("\ufeff", "")
            .strip()

            for column in self.df.columns
        ]

        required_columns = {
            "sample_id",
            "image_path",
            "ground_truth"
        }

        missing = (
            required_columns
            - set(
                self.df.columns
            )
        )

        if missing:

            raise ValueError(
                f"Missing dataset columns: "
                f"{missing}"
            )

        (
            self.charset_data,
            self.char_to_index,
            self.index_to_char,
            self.blank_index
        ) = load_charset()


    def __len__(self):

        return len(
            self.df
        )


    def encode_text(
        self,
        text
    ):

        text = normalize_text(
            text
        )

        encoded = []

        for character in text:

            if character not in self.char_to_index:

                raise ValueError(
                    f"Unknown OCR character: "
                    f"{repr(character)}"
                )

            encoded.append(
                self.char_to_index[
                    character
                ]
            )

        return torch.tensor(
            encoded,
            dtype=torch.long
        )


    def decode_target(
        self,
        encoded
    ):

        characters = []

        for index in encoded:

            index = int(
                index
            )

            if index == self.blank_index:
                continue

            character = (
                self.index_to_char.get(
                    index,
                    ""
                )
            )

            characters.append(
                character
            )

        return "".join(
            characters
        )


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        sample_id = normalize_text(
            row[
                "sample_id"
            ]
        )

        relative_image_path = normalize_text(
            row[
                "image_path"
            ]
        )

        ground_truth = normalize_text(
            row[
                "ground_truth"
            ]
        )

        image_path = (
            OCR_ROOT
            / relative_image_path
        )

        image_tensor = (
            preprocess_image(
                image_path
            )
        )

        encoded_text = (
            self.encode_text(
                ground_truth
            )
        )

        return {
            "sample_id":
                sample_id,

            "image":
                image_tensor,

            "text":
                ground_truth,

            "encoded_text":
                encoded_text,

            "target_length":
                len(
                    encoded_text
                ),

            "image_path":
                str(
                    image_path
                )
        }


# ============================================================
# CTC COLLATE FUNCTION
# ============================================================

def ocr_collate_fn(
    batch
):

    images = torch.stack(
        [
            item["image"]
            for item in batch
        ]
    )

    sample_ids = [
        item["sample_id"]
        for item in batch
    ]

    texts = [
        item["text"]
        for item in batch
    ]

    image_paths = [
        item["image_path"]
        for item in batch
    ]

    target_lengths = torch.tensor(
        [
            item[
                "target_length"
            ]
            for item in batch
        ],
        dtype=torch.long
    )

    targets = torch.cat(
        [
            item[
                "encoded_text"
            ]
            for item in batch
        ]
    )

    return {
        "sample_ids":
            sample_ids,

        "images":
            images,

        "texts":
            texts,

        "targets":
            targets,

        "target_lengths":
            target_lengths,

        "image_paths":
            image_paths
    }
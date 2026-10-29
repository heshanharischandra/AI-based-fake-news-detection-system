
from pathlib import Path
import argparse
import json

import torch

from ocr_crnn_model import CRNN
from ocr_dataset import preprocess_image
from evaluate_ocr_crnn import greedy_ctc_decode


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_crop_jitter_v3"
    / "last_crop_jitter_v3_model.pt"
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
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    index_to_char = {
        int(index): character
        for index, character
        in data["index_to_char"].items()
    }

    blank_index = int(
        data["ctc_blank_index"]
    )

    return (
        index_to_char,
        blank_index
    )


# ============================================================
# OCR V2 ENGINE
# ============================================================

class OCRV2Inference:

    def __init__(
        self,
        model_path=MODEL_PATH,
        device=None
    ):

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"V2 OCR model not found:\n"
                f"{self.model_path}"
            )

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        (
            self.index_to_char,
            self.blank_index
        ) = load_charset()

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        num_classes = int(
            checkpoint[
                "num_classes"
            ]
        )

        hidden_size = int(
            checkpoint.get(
                "hidden_size",
                256
            )
        )

        self.model = CRNN(
            num_classes=num_classes,
            hidden_size=hidden_size
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self.checkpoint_epoch = (
            checkpoint.get(
                "epoch",
                "unknown"
            )
        )

        self.validation_cer = (
            checkpoint.get(
                "validation_cer",
                None
            )
        )

        self.validation_wer = (
            checkpoint.get(
                "validation_wer",
                None
            )
        )

    # ========================================================
    # PREDICT ONE LINE
    # ========================================================

    def predict_line(
        self,
        image_path
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # IMPORTANT
        # Use EXACT SAME preprocessing as training/evaluation
        #
        # preprocess_image returns:
        # [1, H, W]
        #
        # Add batch dimension:
        # [1, 1, H, W]
        # ----------------------------------------------------

        image_tensor = preprocess_image(
            image_path
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        with torch.no_grad():

            log_probs = self.model(
                image_tensor
            )

        # ----------------------------------------------------
        # Use EXACT SAME decoder as evaluation
        # ----------------------------------------------------

        predictions = greedy_ctc_decode(
            log_probs,
            self.index_to_char,
            self.blank_index
        )

        prediction = (
            predictions[0]
            .strip()
        )

        return {
            "status":
                "success",

            "image_path":
                str(
                    image_path
                ),

            "text":
                prediction,

            "model":
                "CRNN-CTC Expanded V2",

            "checkpoint_epoch":
                self.checkpoint_epoch,

            "validation_cer":
                self.validation_cer,

            "validation_wer":
                self.validation_wer,

            "device":
                str(
                    self.device
                )
        }


# ============================================================
# SHARED MODEL INSTANCE
# ============================================================

_ENGINE = None


def get_ocr_v2_engine():

    global _ENGINE

    if _ENGINE is None:

        _ENGINE = OCRV2Inference()

    return _ENGINE


# ============================================================
# SIMPLE FUNCTION
# ============================================================

def predict_line(
    image_path
):

    engine = get_ocr_v2_engine()

    return engine.predict_line(
        image_path
    )


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Expanded V2 Sinhala OCR "
            "on one cropped text-line image."
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to a cropped Sinhala "
            "text-line image."
        )
    )

    args = parser.parse_args()

    print(
        "=" * 72
    )

    print(
        "EXPANDED V2 SINHALA OCR INFERENCE"
    )

    print(
        "=" * 72
    )

    engine = get_ocr_v2_engine()

    print(
        f"\nDevice: "
        f"{engine.device}"
    )

    if (
        engine.device.type
        == "cuda"
    ):

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    print(
        "\nModel:"
    )

    print(
        engine.model_path
    )

    print(
        f"\nCheckpoint epoch: "
        f"{engine.checkpoint_epoch}"
    )

    if (
        engine.validation_cer
        is not None
    ):

        print(
            f"Validation CER: "
            f"{engine.validation_cer:.4f}"
        )

    if (
        engine.validation_wer
        is not None
    ):

        print(
            f"Validation WER: "
            f"{engine.validation_wer:.4f}"
        )

    print(
        "\nInput image:"
    )

    print(
        args.image
    )

    result = engine.predict_line(
        args.image
    )

    print(
        "\n" + "-" * 72
    )

    print(
        "OCR RESULT"
    )

    print(
        "-" * 72
    )

    print(
        "\nPredicted Text:"
    )

    print(
        result[
            "text"
        ]
    )

    print(
        "\nStatus:",
        result[
            "status"
        ]
    )


if __name__ == "__main__":

    main()


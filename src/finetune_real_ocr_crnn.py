from pathlib import Path
from datetime import datetime

import json
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ocr_dataset import (
    OCRDataset,
    ocr_collate_fn
)

from ocr_crnn_model import CRNN

from evaluate_ocr_crnn import (
    greedy_ctc_decode,
    levenshtein_distance
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

REAL_SPLIT_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits"
)

TRAIN_CSV = (
    REAL_SPLIT_DIR
    / "real_finetune_train.csv"
)

VALIDATION_CSV = (
    REAL_SPLIT_DIR
    / "real_finetune_validation.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


# Pretrained synthetic model
SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_full"
    / "best_crnn_model.pt"
)


# New fine-tuned model folder
OUTPUT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_real_finetuned"
)

OUTPUT_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUTPUT_MODEL_DIR
    / "best_real_finetuned_model.pt"
)

LAST_MODEL_PATH = (
    OUTPUT_MODEL_DIR
    / "last_real_finetuned_model.pt"
)

HISTORY_PATH = (
    OUTPUT_MODEL_DIR
    / "real_finetuning_history.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

EPOCHS = 30

BATCH_SIZE = 4

# Much smaller than original synthetic training LR
LEARNING_RATE = 0.0001

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 5.0

NUM_WORKERS = 0

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 8


# ============================================================
# 3. RANDOM SEED
# ============================================================

def set_random_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# 4. LOAD CHARSET
# ============================================================

def load_charset():

    if not CHARSET_JSON.exists():

        raise FileNotFoundError(
            f"Charset file not found:\n"
            f"{CHARSET_JSON}"
        )

    with open(
        CHARSET_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    index_to_char = {

        int(index): character

        for index, character
        in data[
            "index_to_char"
        ].items()
    }

    blank_index = int(
        data[
            "ctc_blank_index"
        ]
    )

    return (
        index_to_char,
        blank_index
    )


# ============================================================
# 5. DATA LOADERS
# ============================================================

def create_data_loaders():

    train_dataset = OCRDataset(
        TRAIN_CSV
    )

    validation_dataset = OCRDataset(
        VALIDATION_CSV
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    )


# ============================================================
# 6. CTC INPUT LENGTHS
# ============================================================

def create_input_lengths(
    log_probs
):

    time_steps = (
        log_probs.size(0)
    )

    batch_size = (
        log_probs.size(1)
    )

    return torch.full(
        (
            batch_size,
        ),
        time_steps,
        dtype=torch.long
    )


# ============================================================
# 7. KEEP BATCHNORM STABLE
# ============================================================

def freeze_batchnorm_statistics(
    model
):

    for module in model.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            module.eval()


# ============================================================
# 8. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    # Only 24 real samples are available.
    # Keep BatchNorm running statistics stable.
    freeze_batchnorm_statistics(
        model
    )

    running_loss = 0.0

    valid_batches = 0

    for batch in loader:

        images = (
            batch[
                "images"
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        targets = batch[
            "targets"
        ]

        target_lengths = batch[
            "target_lengths"
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        log_probs = model(
            images
        )

        input_lengths = (
            create_input_lengths(
                log_probs
            )
        )

        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )

        if not torch.isfinite(
            loss
        ):

            print(
                "WARNING: Non-finite "
                "training loss skipped."
            )

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

        valid_batches += 1

    if valid_batches == 0:

        return math.inf

    return (
        running_loss
        / valid_batches
    )


# ============================================================
# 9. VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
    index_to_char,
    blank_index
):

    model.eval()

    running_loss = 0.0

    valid_batches = 0

    total_character_errors = 0
    total_reference_characters = 0

    predictions_for_display = []

    with torch.no_grad():

        for batch in loader:

            images = (
                batch[
                    "images"
                ]
                .to(
                    device,
                    non_blocking=True
                )
            )

            targets = batch[
                "targets"
            ]

            target_lengths = batch[
                "target_lengths"
            ]

            references = batch[
                "texts"
            ]

            log_probs = model(
                images
            )

            input_lengths = (
                create_input_lengths(
                    log_probs
                )
            )

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

            if torch.isfinite(
                loss
            ):

                running_loss += (
                    loss.item()
                )

                valid_batches += 1

            predictions = (
                greedy_ctc_decode(
                    log_probs,
                    index_to_char,
                    blank_index
                )
            )

            for reference, prediction in zip(
                references,
                predictions
            ):

                distance = (
                    levenshtein_distance(
                        list(reference),
                        list(prediction)
                    )
                )

                total_character_errors += (
                    distance
                )

                total_reference_characters += (
                    len(reference)
                )

                predictions_for_display.append(
                    (
                        reference,
                        prediction
                    )
                )

    validation_loss = (

        running_loss
        / valid_batches

        if valid_batches > 0

        else math.inf
    )

    validation_cer = (

        total_character_errors
        / total_reference_characters

        if total_reference_characters > 0

        else math.inf
    )

    return (
        validation_loss,
        validation_cer,
        predictions_for_display
    )


# ============================================================
# 10. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    validation_loss,
    validation_cer,
    num_classes,
    hidden_size
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "validation_cer":
            validation_cer,

        "num_classes":
            num_classes,

        "hidden_size":
            hidden_size,

        "learning_rate":
            optimizer.param_groups[
                0
            ][
                "lr"
            ],

        "fine_tuned_from":
            str(
                SOURCE_MODEL_PATH
            ),

        "created_at":
            datetime.now().isoformat()
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL-DOMAIN SINHALA OCR FINE-TUNING"
    )

    print(
        "=" * 72
    )

    set_random_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    ) = create_data_loaders()

    print(
        f"\nReal train samples      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Real validation samples : "
        f"{len(validation_dataset)}"
    )

    print(
        f"Batch size              : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Train batches/epoch     : "
        f"{len(train_loader)}"
    )

    # --------------------------------------------------------
    # Load source checkpoint
    # --------------------------------------------------------

    if not SOURCE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Source model not found:\n"
            f"{SOURCE_MODEL_PATH}"
        )

    checkpoint = torch.load(
        SOURCE_MODEL_PATH,
        map_location=device,
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

    model = CRNN(
        num_classes=num_classes,
        hidden_size=hidden_size
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    print(
        "\nLoaded pretrained synthetic model:"
    )

    print(
        SOURCE_MODEL_PATH
    )

    print(
        f"Source checkpoint epoch : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Source validation loss  : "
        f"{checkpoint['validation_loss']:.4f}"
    )

    # --------------------------------------------------------
    # Charset
    # --------------------------------------------------------

    (
        index_to_char,
        blank_index
    ) = load_charset()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CTCLoss(
        blank=blank_index,
        reduction="mean",
        zero_infinity=True
    )

    # --------------------------------------------------------
    # NEW optimizer for real fine-tuning
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    )

    # --------------------------------------------------------
    # BEFORE fine-tuning validation
    # --------------------------------------------------------

    (
        baseline_loss,
        baseline_cer,
        baseline_predictions
    ) = validate(
        model,
        validation_loader,
        criterion,
        device,
        index_to_char,
        blank_index
    )

    print(
        "\n" + "-" * 72
    )

    print(
        "BEFORE FINE-TUNING"
    )

    print(
        "-" * 72
    )

    print(
        f"Validation Loss : "
        f"{baseline_loss:.4f}"
    )

    print(
        f"Validation CER  : "
        f"{baseline_cer:.4f}"
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {

        "configuration": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "real_train_samples":
                len(
                    train_dataset
                ),

            "real_validation_samples":
                len(
                    validation_dataset
                ),

            "source_model":
                str(
                    SOURCE_MODEL_PATH
                )
        },

        "before_finetuning": {

            "validation_loss":
                baseline_loss,

            "validation_cer":
                baseline_cer
        },

        "epochs": []
    }

    # --------------------------------------------------------
    # Best tracking
    # --------------------------------------------------------

    best_cer = (
        baseline_cer
    )

    best_validation_loss = (
        baseline_loss
    )

    epochs_without_cer_improvement = 0

    training_start = (
        time.time()
    )

    # ========================================================
    # FINE-TUNING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = (
            time.time()
        )

        print(
            "\n" + "-" * 72
        )

        print(
            f"FINE-TUNE EPOCH "
            f"{epoch}/{EPOCHS}"
        )

        print(
            "-" * 72
        )

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )
        )

        (
            validation_loss,
            validation_cer,
            validation_predictions
        ) = validate(
            model,
            validation_loader,
            criterion,
            device,
            index_to_char,
            blank_index
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss : "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation CER  : "
            f"{validation_cer:.4f}"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch Time      : "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Show first validation prediction
        # ----------------------------------------------------

        if validation_predictions:

            reference, prediction = (
                validation_predictions[
                    0
                ]
            )

            print(
                "\nExample validation prediction:"
            )

            print(
                "GT  :",
                reference
            )

            print(
                "PRED:",
                prediction
            )

        # ----------------------------------------------------
        # Save last
        # ----------------------------------------------------

        save_checkpoint(
            LAST_MODEL_PATH,
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            validation_loss,
            validation_cer,
            num_classes,
            hidden_size
        )

        # ----------------------------------------------------
        # Best model is selected by CER
        # ----------------------------------------------------

        improved = False

        if (
            validation_cer
            < best_cer
        ):

            improved = True

        elif (
            validation_cer
            == best_cer
            and validation_loss
            < best_validation_loss
        ):

            improved = True

        if improved:

            best_cer = (
                validation_cer
            )

            best_validation_loss = (
                validation_loss
            )

            epochs_without_cer_improvement = 0

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                validation_loss,
                validation_cer,
                num_classes,
                hidden_size
            )

            print(
                "\nBest real-domain "
                "checkpoint updated."
            )

        else:

            epochs_without_cer_improvement += 1

            print(
                "\nNo CER improvement."
            )

            print(
                "Early-stop counter: "
                f"{epochs_without_cer_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history[
            "epochs"
        ].append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            "validation_cer":
                validation_cer,

            "learning_rate":
                current_lr,

            "epoch_seconds":
                epoch_time
        })

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                ensure_ascii=False,
                indent=4
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_cer_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEARLY STOPPING ACTIVATED"
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    total_time = (
        time.time()
        - training_start
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "REAL-DOMAIN FINE-TUNING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nBefore CER : "
        f"{baseline_cer:.4f}"
    )

    print(
        f"Best CER   : "
        f"{best_cer:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Total time : "
        f"{total_time:.1f}s"
    )

    print(
        "\nBest fine-tuned model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nHistory:"
    )

    print(
        HISTORY_PATH
    )
from pathlib import Path
from datetime import datetime

import csv
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ocr_dataset import (
    OCRDataset,
    ocr_collate_fn,
)

from ocr_crnn_model import CRNN

from evaluate_ocr_crnn import (
    greedy_ctc_decode,
    levenshtein_distance,
)


# ============================================================
# VERSION
# ============================================================

TRAINING_VERSION = (
    "CLEAN REAL OCR V4 FINE-TUNING"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)


# ============================================================
# CLEAN V4 TRAINING DATA
#
# 76 clean original train lines
# +
# 912 train-only augmentations
# =
# 988 training samples
# ============================================================

TRAIN_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "augmentation_v4"
    / "metadata"
    / "clean_real_train_combined_v4.csv"
)


# ============================================================
# CLEAN VALIDATION DATA
#
# 23 untouched real validation lines
# 6 completely held-out sources
# ============================================================

VALIDATION_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "metadata"
    / "splits"
    / "clean_real_validation_v1.csv"
)


# ============================================================
# CHARSET
# ============================================================

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


# ============================================================
# SOURCE MODEL
#
# IMPORTANT:
# Start from synthetic-only checkpoint.
#
# DO NOT start from V2 or V3 real-fine-tuned model.
#
# This keeps the V4 experiment methodologically clean.
# ============================================================

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_full"
    / "best_crnn_model.pt"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_clean_v4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_clean_v4_model.pt"
)

LAST_MODEL_PATH = (
    OUTPUT_DIR
    / "last_clean_v4_model.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "clean_v4_training_history.json"
)


# ============================================================
# EXPECTED DATASET INFORMATION
# ============================================================

EXPECTED_TRAIN_SAMPLES = 988

EXPECTED_VALIDATION_SAMPLES = 23

EXPECTED_TRAIN_SOURCES = 24

EXPECTED_VALIDATION_SOURCES = 6


# ============================================================
# TRAINING CONFIGURATION
#
# Preserve V3 hyperparameters for a fair experiment.
# ============================================================

EPOCHS = 20

BATCH_SIZE = 4

LEARNING_RATE = 0.00005

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 5.0

NUM_WORKERS = 0

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 6

SCHEDULER_PATIENCE = 2


# ============================================================
# RANDOM SEED
# ============================================================

def set_random_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# LOAD CSV FOR PREFLIGHT VALIDATION
# ============================================================

def load_csv_records(path):

    if not path.exists():

        raise FileNotFoundError(
            f"CSV file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# PREFLIGHT DATASET VALIDATION
#
# This does not replace OCRDataset.
# It only prevents accidentally training with a bad manifest.
# ============================================================

def validate_manifests():

    train_records = (
        load_csv_records(
            TRAIN_CSV
        )
    )

    validation_records = (
        load_csv_records(
            VALIDATION_CSV
        )
    )

    print()
    print(
        "-" * 72
    )

    print(
        "V4 DATASET PREFLIGHT CHECK"
    )

    print(
        "-" * 72
    )

    print(
        "Training manifest rows:",
        len(
            train_records
        )
    )

    print(
        "Validation manifest rows:",
        len(
            validation_records
        )
    )

    # --------------------------------------------------------
    # BLANK LABEL CHECK
    # --------------------------------------------------------

    blank_train_labels = [

        str(
            row.get(
                "sample_id",
                ""
            )
        ).strip()

        for row in train_records

        if not str(
            row.get(
                "ground_truth",
                ""
            )
        ).strip()
    ]

    blank_validation_labels = [

        str(
            row.get(
                "sample_id",
                ""
            )
        ).strip()

        for row in validation_records

        if not str(
            row.get(
                "ground_truth",
                ""
            )
        ).strip()
    ]

    if blank_train_labels:

        raise RuntimeError(
            "Blank TRAIN labels found:\n"
            + "\n".join(
                blank_train_labels
            )
        )

    if blank_validation_labels:

        raise RuntimeError(
            "Blank VALIDATION labels found:\n"
            + "\n".join(
                blank_validation_labels
            )
        )

    # --------------------------------------------------------
    # SOURCE-LEVEL LEAKAGE CHECK
    # --------------------------------------------------------

    train_sources = {

        str(
            row.get(
                "source_id",
                ""
            )
        ).strip()

        for row in train_records
    }

    validation_sources = {

        str(
            row.get(
                "source_id",
                ""
            )
        ).strip()

        for row in validation_records
    }

    source_overlap = (
        train_sources
        & validation_sources
    )

    if source_overlap:

        raise RuntimeError(
            "SOURCE LEAKAGE DETECTED:\n"
            + "\n".join(
                sorted(
                    source_overlap
                )
            )
        )

    # --------------------------------------------------------
    # SAMPLE-ID LEAKAGE CHECK
    # --------------------------------------------------------

    train_ids = {

        str(
            row.get(
                "sample_id",
                ""
            )
        ).strip()

        for row in train_records
    }

    validation_ids = {

        str(
            row.get(
                "sample_id",
                ""
            )
        ).strip()

        for row in validation_records
    }

    sample_overlap = (
        train_ids
        & validation_ids
    )

    if sample_overlap:

        raise RuntimeError(
            "SAMPLE-ID LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # PARENT LEAKAGE CHECK
    #
    # Augmented train samples contain parent_sample_id.
    # Make sure no validation sample ID is an augmented parent.
    # --------------------------------------------------------

    train_parent_ids = {

        str(
            row.get(
                "parent_sample_id",
                ""
            )
        ).strip()

        for row in train_records

        if str(
            row.get(
                "parent_sample_id",
                ""
            )
        ).strip()
    }

    parent_validation_overlap = (
        train_parent_ids
        & validation_ids
    )

    if parent_validation_overlap:

        raise RuntimeError(
            "AUGMENTATION PARENT LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # EXPECTED COUNT WARNINGS
    # --------------------------------------------------------

    if (
        len(train_records)
        != EXPECTED_TRAIN_SAMPLES
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected {EXPECTED_TRAIN_SAMPLES} "
            f"training samples, but found "
            f"{len(train_records)}."
        )

    if (
        len(validation_records)
        != EXPECTED_VALIDATION_SAMPLES
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected {EXPECTED_VALIDATION_SAMPLES} "
            f"validation samples, but found "
            f"{len(validation_records)}."
        )

    if (
        len(train_sources)
        != EXPECTED_TRAIN_SOURCES
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected {EXPECTED_TRAIN_SOURCES} "
            f"training sources, but found "
            f"{len(train_sources)}."
        )

    if (
        len(validation_sources)
        != EXPECTED_VALIDATION_SOURCES
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected {EXPECTED_VALIDATION_SOURCES} "
            f"validation sources, but found "
            f"{len(validation_sources)}."
        )

    print()
    print(
        "Training sources:",
        len(
            train_sources
        )
    )

    print(
        "Validation sources:",
        len(
            validation_sources
        )
    )

    print(
        "Source leakage:",
        len(
            source_overlap
        )
    )

    print(
        "Sample leakage:",
        len(
            sample_overlap
        )
    )

    print(
        "Parent leakage:",
        len(
            parent_validation_overlap
        )
    )

    print(
        "Blank train labels:",
        len(
            blank_train_labels
        )
    )

    print(
        "Blank validation labels:",
        len(
            blank_validation_labels
        )
    )

    print()
    print(
        "PREFLIGHT VALIDATION PASSED"
    )

    return (
        train_records,
        validation_records,
        train_sources,
        validation_sources,
    )


# ============================================================
# LOAD CHARSET
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
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    index_to_char = {

        int(index):
            character

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
        blank_index,
    )


# ============================================================
# DATA LOADERS
#
# Same OCRDataset / preprocessing pipeline used by V3.
# ============================================================

def create_data_loaders():

    train_dataset = OCRDataset(
        TRAIN_CSV
    )

    validation_dataset = OCRDataset(
        VALIDATION_CSV
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn,
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    )


# ============================================================
# CTC INPUT LENGTHS
# ============================================================

def create_input_lengths(
    log_probs
):

    time_steps = (
        log_probs.size(
            0
        )
    )

    batch_size = (
        log_probs.size(
            1
        )
    )

    return torch.full(
        (
            batch_size,
        ),
        time_steps,
        dtype=torch.long,
    )


# ============================================================
# FREEZE BATCHNORM RUNNING STATISTICS
#
# Preserve V3 behaviour.
# ============================================================

def freeze_batchnorm_statistics(
    model
):

    for module in model.modules():

        if isinstance(
            module,
            nn.BatchNorm2d,
        ):

            module.eval()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
):

    model.train()

    freeze_batchnorm_statistics(
        model
    )

    total_loss = 0.0

    valid_batches = 0

    total_batches = len(
        loader
    )

    epoch_start = (
        time.time()
    )

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):

        images = (
            batch[
                "images"
            ]
            .to(
                device,
                non_blocking=True,
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
            target_lengths,
        )

        if not torch.isfinite(
            loss
        ):

            print(
                f"WARNING: invalid loss "
                f"at batch {batch_number}"
            )

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

        valid_batches += 1

        if (
            batch_number == 1
            or batch_number % 50 == 0
            or batch_number == total_batches
        ):

            average_loss = (
                total_loss
                / valid_batches
            )

            elapsed = (
                time.time()
                - epoch_start
            )

            print(
                f"Epoch {epoch:02d}"
                f" | Batch "
                f"{batch_number}/"
                f"{total_batches}"
                f" | Avg Loss "
                f"{average_loss:.4f}"
                f" | {elapsed:.1f}s"
            )

    if valid_batches == 0:

        return math.inf

    return (
        total_loss
        / valid_batches
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
    index_to_char,
    blank_index,
):

    model.eval()

    total_loss = 0.0

    valid_batches = 0

    total_character_errors = 0

    total_reference_characters = 0

    total_word_errors = 0

    total_reference_words = 0

    exact_matches = 0

    sample_count = 0

    examples = []

    with torch.no_grad():

        for batch in loader:

            images = (
                batch[
                    "images"
                ]
                .to(
                    device,
                    non_blocking=True,
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
                target_lengths,
            )

            if torch.isfinite(
                loss
            ):

                total_loss += (
                    loss.item()
                )

                valid_batches += 1

            predictions = (
                greedy_ctc_decode(
                    log_probs,
                    index_to_char,
                    blank_index,
                )
            )

            for reference, prediction in zip(
                references,
                predictions,
            ):

                sample_count += 1

                # --------------------------------------------
                # CER
                # --------------------------------------------

                character_errors = (
                    levenshtein_distance(
                        list(
                            reference
                        ),
                        list(
                            prediction
                        ),
                    )
                )

                total_character_errors += (
                    character_errors
                )

                total_reference_characters += (
                    len(
                        reference
                    )
                )

                # --------------------------------------------
                # WER
                # --------------------------------------------

                reference_words = (
                    reference.split()
                )

                prediction_words = (
                    prediction.split()
                )

                word_errors = (
                    levenshtein_distance(
                        reference_words,
                        prediction_words,
                    )
                )

                total_word_errors += (
                    word_errors
                )

                total_reference_words += (
                    len(
                        reference_words
                    )
                )

                # --------------------------------------------
                # EXACT MATCH
                # --------------------------------------------

                if prediction == reference:

                    exact_matches += 1

                if len(
                    examples
                ) < 5:

                    examples.append(
                        (
                            reference,
                            prediction,
                        )
                    )

    validation_loss = (

        total_loss
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

    validation_wer = (

        total_word_errors
        / total_reference_words

        if total_reference_words > 0

        else math.inf
    )

    exact_accuracy = (

        exact_matches
        / sample_count

        if sample_count > 0

        else 0.0
    )

    return (
        validation_loss,
        validation_cer,
        validation_wer,
        exact_accuracy,
        exact_matches,
        examples,
    )


# ============================================================
# SAVE CHECKPOINT
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
    validation_wer,
    exact_accuracy,
    num_classes,
    hidden_size,
):

    checkpoint = {

        "training_version":
            TRAINING_VERSION,

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

        "validation_wer":
            validation_wer,

        "exact_accuracy":
            exact_accuracy,

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

        "source_model":
            str(
                SOURCE_MODEL_PATH
            ),

        "training_csv":
            str(
                TRAIN_CSV
            ),

        "validation_csv":
            str(
                VALIDATION_CSV
            ),

        "created_at":
            datetime.now().isoformat(),
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        TRAINING_VERSION
    )

    print(
        "=" * 72
    )

    set_random_seed(
        RANDOM_SEED
    )

    # ========================================================
    # PREFLIGHT
    # ========================================================

    (
        manifest_train_records,
        manifest_validation_records,
        train_sources,
        validation_sources,
    ) = validate_manifests()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    # ========================================================
    # DATASET
    # ========================================================

    (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_data_loaders()

    print()
    print(
        "Training samples      :",
        len(
            train_dataset
        )
    )

    print(
        "Validation samples    :",
        len(
            validation_dataset
        )
    )

    print(
        "Batch size            :",
        BATCH_SIZE
    )

    print(
        "Train batches/epoch   :",
        len(
            train_loader
        )
    )

    print(
        "Validation batches    :",
        len(
            validation_loader
        )
    )

    if (
        len(train_dataset)
        != len(
            manifest_train_records
        )
    ):

        raise RuntimeError(
            "OCRDataset training count does not match "
            "training manifest count."
        )

    if (
        len(validation_dataset)
        != len(
            manifest_validation_records
        )
    ):

        raise RuntimeError(
            "OCRDataset validation count does not match "
            "validation manifest count."
        )

    # ========================================================
    # LOAD CLEAN SYNTHETIC-ONLY CHECKPOINT
    # ========================================================

    if not SOURCE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Source model not found:\n"
            f"{SOURCE_MODEL_PATH}"
        )

    checkpoint = torch.load(
        SOURCE_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    num_classes = int(
        checkpoint[
            "num_classes"
        ]
    )

    hidden_size = int(
        checkpoint.get(
            "hidden_size",
            256,
        )
    )

    model = CRNN(
        num_classes=num_classes,
        hidden_size=hidden_size,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    print()
    print(
        "Loaded CLEAN synthetic-only checkpoint:"
    )

    print(
        SOURCE_MODEL_PATH
    )

    print(
        "Source epoch          :",
        checkpoint[
            "epoch"
        ]
    )

    print(
        "Source validation loss:",
        f"{checkpoint['validation_loss']:.4f}",
    )

    print(
        "Number of classes     :",
        num_classes
    )

    print(
        "Hidden size           :",
        hidden_size
    )

    # ========================================================
    # CHARSET
    # ========================================================

    (
        index_to_char,
        blank_index,
    ) = load_charset()

    print()
    print(
        "Charset classes:",
        len(
            index_to_char
        )
    )

    print(
        "CTC blank index:",
        blank_index
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CTCLoss(
        blank=blank_index,
        reduction="mean",
        zero_infinity=True,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # SCHEDULER
    #
    # Same as V3: CER is primary development metric.
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=SCHEDULER_PATIENCE,
            min_lr=1e-6,
        )
    )

    # ========================================================
    # BASELINE
    #
    # Synthetic-only model on NEW cleaned 23-line validation.
    # ========================================================

    (
        baseline_loss,
        baseline_cer,
        baseline_wer,
        baseline_exact_accuracy,
        baseline_exact_matches,
        baseline_examples,
    ) = validate(
        model,
        validation_loader,
        criterion,
        device,
        index_to_char,
        blank_index,
    )

    print()
    print(
        "-" * 72
    )

    print(
        "BEFORE CLEAN V4 FINE-TUNING"
    )

    print(
        "-" * 72
    )

    print(
        "Validation Loss :",
        f"{baseline_loss:.4f}",
    )

    print(
        "Validation CER  :",
        f"{baseline_cer:.4f}",
    )

    print(
        "Validation WER  :",
        f"{baseline_wer:.4f}",
    )

    print(
        "Exact matches   :",
        f"{baseline_exact_matches}/"
        f"{len(validation_dataset)}",
    )

    print(
        "Exact accuracy  :",
        f"{baseline_exact_accuracy * 100:.2f}%",
    )

    print()
    print(
        "Baseline examples:"
    )

    for example_number, (
        reference,
        prediction,
    ) in enumerate(
        baseline_examples,
        start=1,
    ):

        print()
        print(
            f"Example {example_number}:"
        )

        print(
            "GT  :",
            reference
        )

        print(
            "PRED:",
            prediction
        )

    # ========================================================
    # HISTORY
    # ========================================================

    history = {

        "training_version":
            TRAINING_VERSION,

        "configuration": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "initial_learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "grad_clip":
                GRAD_CLIP,

            "random_seed":
                RANDOM_SEED,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "scheduler_patience":
                SCHEDULER_PATIENCE,

            "training_samples":
                len(
                    train_dataset
                ),

            "validation_samples":
                len(
                    validation_dataset
                ),

            "training_sources":
                len(
                    train_sources
                ),

            "validation_sources":
                len(
                    validation_sources
                ),

            "source_model":
                str(
                    SOURCE_MODEL_PATH
                ),

            "training_csv":
                str(
                    TRAIN_CSV
                ),

            "validation_csv":
                str(
                    VALIDATION_CSV
                ),
        },

        "before_finetuning": {

            "validation_loss":
                baseline_loss,

            "validation_cer":
                baseline_cer,

            "validation_wer":
                baseline_wer,

            "exact_accuracy":
                baseline_exact_accuracy,

            "exact_matches":
                baseline_exact_matches,
        },

        "epochs": [],
    }

    # ========================================================
    # BEST MODEL TRACKING
    #
    # Priority:
    # 1. CER
    # 2. WER
    # 3. Validation loss
    # ========================================================

    best_cer = (
        baseline_cer
    )

    best_wer = (
        baseline_wer
    )

    best_validation_loss = (
        baseline_loss
    )

    best_exact_accuracy = (
        baseline_exact_accuracy
    )

    best_epoch = 0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Save epoch-0 baseline checkpoint.
    # --------------------------------------------------------

    save_checkpoint(
        BEST_MODEL_PATH,
        model,
        optimizer,
        scheduler,
        0,
        None,
        baseline_loss,
        baseline_cer,
        baseline_wer,
        baseline_exact_accuracy,
        num_classes,
        hidden_size,
    )

    print()
    print(
        "Epoch 0 baseline checkpoint saved:"
    )

    print(
        BEST_MODEL_PATH
    )

    training_start = (
        time.time()
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print(
            "=" * 72
        )

        print(
            f"CLEAN V4 FINE-TUNE EPOCH "
            f"{epoch}/{EPOCHS}"
        )

        print(
            "=" * 72
        )

        epoch_start = (
            time.time()
        )

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
            )
        )

        (
            validation_loss,
            validation_cer,
            validation_wer,
            exact_accuracy,
            exact_matches,
            examples,
        ) = validate(
            model,
            validation_loader,
            criterion,
            device,
            index_to_char,
            blank_index,
        )

        # ----------------------------------------------------
        # Scheduler follows CER.
        # ----------------------------------------------------

        scheduler.step(
            validation_cer
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        print()
        print(
            "Epoch Summary"
        )

        print(
            "Train Loss      :",
            f"{train_loss:.4f}",
        )

        print(
            "Validation Loss :",
            f"{validation_loss:.4f}",
        )

        print(
            "Validation CER  :",
            f"{validation_cer:.4f}",
        )

        print(
            "Validation WER  :",
            f"{validation_wer:.4f}",
        )

        print(
            "Exact matches   :",
            f"{exact_matches}/"
            f"{len(validation_dataset)}",
        )

        print(
            "Exact Accuracy  :",
            f"{exact_accuracy * 100:.2f}%",
        )

        print(
            "Learning Rate   :",
            f"{current_lr:.8f}",
        )

        print(
            "Epoch Time      :",
            f"{epoch_seconds:.1f}s",
        )

        # ----------------------------------------------------
        # SHOW EXAMPLES
        # ----------------------------------------------------

        for example_number, (
            reference,
            prediction,
        ) in enumerate(
            examples,
            start=1,
        ):

            print()
            print(
                f"Example {example_number}:"
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
        # ALWAYS SAVE LAST MODEL
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
            validation_wer,
            exact_accuracy,
            num_classes,
            hidden_size,
        )

        # ====================================================
        # BEST MODEL SELECTION
        # ====================================================

        improved = False

        tolerance = 1e-12

        if (
            validation_cer
            < best_cer - tolerance
        ):

            improved = True

        elif (
            abs(
                validation_cer
                - best_cer
            )
            <= tolerance
            and validation_wer
            < best_wer - tolerance
        ):

            improved = True

        elif (
            abs(
                validation_cer
                - best_cer
            )
            <= tolerance
            and abs(
                validation_wer
                - best_wer
            )
            <= tolerance
            and validation_loss
            < best_validation_loss
        ):

            improved = True

        if improved:

            best_cer = (
                validation_cer
            )

            best_wer = (
                validation_wer
            )

            best_validation_loss = (
                validation_loss
            )

            best_exact_accuracy = (
                exact_accuracy
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                validation_loss,
                validation_cer,
                validation_wer,
                exact_accuracy,
                num_classes,
                hidden_size,
            )

            print()
            print(
                "BEST CLEAN V4 CHECKPOINT UPDATED ✅"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "No validation improvement."
            )

            print(
                "Early-stop counter:",
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}",
            )

        # ====================================================
        # HISTORY
        # ====================================================

        history[
            "epochs"
        ].append(
            {

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "validation_loss":
                    validation_loss,

                "validation_cer":
                    validation_cer,

                "validation_wer":
                    validation_wer,

                "exact_accuracy":
                    exact_accuracy,

                "exact_matches":
                    exact_matches,

                "learning_rate":
                    current_lr,

                "epoch_seconds":
                    epoch_seconds,
            }
        )

        history[
            "current_best"
        ] = {

            "epoch":
                best_epoch,

            "cer":
                best_cer,

            "wer":
                best_wer,

            "validation_loss":
                best_validation_loss,

            "exact_accuracy":
                best_exact_accuracy,
        }

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                ensure_ascii=False,
                indent=4,
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "EARLY STOPPING ACTIVATED"
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    total_training_time = (
        time.time()
        - training_start
    )

    print()
    print(
        "=" * 72
    )

    print(
        "CLEAN REAL OCR V4 FINE-TUNING COMPLETE"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "Baseline CER :",
        f"{baseline_cer:.4f}",
    )

    print(
        "Best CER     :",
        f"{best_cer:.4f}",
    )

    print(
        "Baseline WER :",
        f"{baseline_wer:.4f}",
    )

    print(
        "Best WER     :",
        f"{best_wer:.4f}",
    )

    print(
        "Baseline exact:",
        f"{baseline_exact_accuracy * 100:.2f}%",
    )

    print(
        "Best exact   :",
        f"{best_exact_accuracy * 100:.2f}%",
    )

    print(
        "Best epoch   :",
        best_epoch,
    )

    print(
        "Best loss    :",
        f"{best_validation_loss:.4f}",
    )

    print(
        "Training time:",
        f"{total_training_time:.1f}s",
    )

    print()
    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()
    print(
        "Last model:"
    )

    print(
        LAST_MODEL_PATH
    )

    print()
    print(
        "History:"
    )

    print(
        HISTORY_PATH
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The 23-line cleaned validation set "
        "was never augmented."
    )

    print(
        "A9 was NOT used during this "
        "training process."
    )

    print(
        "For a fair V3 vs V4 comparison, "
        "evaluate BOTH models on the same "
        "clean 23-line validation set."
    )

    print(
        "=" * 72
    )
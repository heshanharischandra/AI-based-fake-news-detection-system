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
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

TRAIN_CSV = (
    OCR_ROOT
    / "real_training_expansion_v5"
    / "augmentation_v5"
    / "metadata"
    / "v5_train_combined_augmented.csv"
)

VALIDATION_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "metadata"
    / "splits"
    / "clean_real_validation_v1.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


# ============================================================
# SOURCE MODEL
#
# V3 is currently our strongest custom CRNN.
#
# We intentionally use the usable V3 LAST checkpoint because
# the previous V3 run finished at its strongest epoch and the
# separate best file was not available.
#
# V5 validation below will independently evaluate this model
# on the CLEAN 23-line validation set before any new training.
# ============================================================

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_crop_jitter_v3"
    / "last_crop_jitter_v3_model.pt"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_real_v5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_real_v5_model.pt"
)

LAST_MODEL_PATH = (
    OUTPUT_DIR
    / "last_real_v5_model.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "real_v5_training_history.json"
)


# ============================================================
# EXPECTED DATASET COUNTS
# ============================================================

EXPECTED_TRAIN_COUNT = 1573
EXPECTED_VALIDATION_COUNT = 23

EXPECTED_ORIGINAL_REAL_LINES = 121
EXPECTED_AUGMENTED_LINES = 1452


# ============================================================
# TRAINING CONFIGURATION
#
# V3 used 5e-5 from synthetic-only initialization.
#
# V5 starts from an already real-domain fine-tuned V3 model,
# therefore we use a smaller learning rate to reduce the risk
# of destroying useful V3 features.
# ============================================================

EPOCHS = 15

BATCH_SIZE = 4

LEARNING_RATE = 0.00002

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 5.0

NUM_WORKERS = 0

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 5

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
# LOAD CSV
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ============================================================
# PRE-FLIGHT DATASET CHECK
# ============================================================

def validate_training_metadata():

    train_records = load_csv(
        TRAIN_CSV
    )

    validation_records = load_csv(
        VALIDATION_CSV
    )

    print()
    print("=" * 72)
    print("V5 DATASET PRE-FLIGHT CHECK")
    print("=" * 72)

    print(
        "Training records:",
        len(train_records),
    )

    print(
        "Validation records:",
        len(validation_records),
    )

    if len(train_records) != EXPECTED_TRAIN_COUNT:

        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_COUNT} "
            f"training samples, found "
            f"{len(train_records)}."
        )

    if (
        len(validation_records)
        != EXPECTED_VALIDATION_COUNT
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_COUNT} "
            f"validation samples, found "
            f"{len(validation_records)}."
        )

    sample_ids = set()

    duplicate_ids = []

    blank_train_labels = []

    original_count = 0

    augmented_count = 0

    train_sources = set()

    for record in train_records:

        sample_id = str(
            record.get(
                "sample_id",
                "",
            )
        ).strip()

        source_id = str(
            record.get(
                "source_id",
                "",
            )
        ).strip()

        ground_truth = str(
            record.get(
                "ground_truth",
                "",
            )
        ).strip()

        is_augmented = str(
            record.get(
                "is_augmented",
                "",
            )
        ).strip()

        if sample_id in sample_ids:

            duplicate_ids.append(
                sample_id
            )

        sample_ids.add(
            sample_id
        )

        if not ground_truth:

            blank_train_labels.append(
                sample_id
            )

        if source_id:

            train_sources.add(
                source_id
            )

        if is_augmented == "1":

            augmented_count += 1

        elif is_augmented == "0":

            original_count += 1

        else:

            raise RuntimeError(
                f"{sample_id}: invalid "
                f"is_augmented='{is_augmented}'."
            )

    validation_sources = {
        str(
            record.get(
                "source_id",
                "",
            )
        ).strip()
        for record in validation_records
    }

    validation_blank_labels = []

    for record in validation_records:

        sample_id = str(
            record.get(
                "sample_id",
                "",
            )
        ).strip()

        ground_truth = str(
            record.get(
                "ground_truth",
                "",
            )
        ).strip()

        if not ground_truth:

            validation_blank_labels.append(
                sample_id
            )

    source_overlap = (
        train_sources
        & validation_sources
    )

    print(
        "Original real lines:",
        original_count,
    )

    print(
        "Augmented lines:",
        augmented_count,
    )

    print(
        "Training sources:",
        len(train_sources),
    )

    print(
        "Validation sources:",
        len(validation_sources),
    )

    print(
        "Source leakage:",
        len(source_overlap),
    )

    print(
        "Duplicate training IDs:",
        len(duplicate_ids),
    )

    print(
        "Blank training labels:",
        len(blank_train_labels),
    )

    print(
        "Blank validation labels:",
        len(validation_blank_labels),
    )

    if (
        original_count
        != EXPECTED_ORIGINAL_REAL_LINES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_ORIGINAL_REAL_LINES} "
            f"original real lines, found "
            f"{original_count}."
        )

    if (
        augmented_count
        != EXPECTED_AUGMENTED_LINES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_AUGMENTED_LINES} "
            f"augmented lines, found "
            f"{augmented_count}."
        )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate training sample IDs:\n"
            + "\n".join(
                duplicate_ids
            )
        )

    if blank_train_labels:

        raise RuntimeError(
            "Blank training labels:\n"
            + "\n".join(
                blank_train_labels
            )
        )

    if validation_blank_labels:

        raise RuntimeError(
            "Blank validation labels:\n"
            + "\n".join(
                validation_blank_labels
            )
        )

    if source_overlap:

        raise RuntimeError(
            "Train/validation source leakage:\n"
            + "\n".join(
                sorted(
                    source_overlap
                )
            )
        )

    print()
    print("V5 DATASET PRE-FLIGHT PASSED")
    print("=" * 72)


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
# ============================================================

def create_data_loaders():

    train_dataset = OCRDataset(
        TRAIN_CSV
    )

    validation_dataset = OCRDataset(
        VALIDATION_CSV
    )

    generator = torch.Generator()

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
    log_probs,
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
# ============================================================

def freeze_batchnorm_statistics(
    model,
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

    epoch_start = time.time()

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
                ) < 3:

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

    print("=" * 72)
    print("REAL SINHALA OCR V5 FINE-TUNING")
    print("=" * 72)

    set_random_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # PRE-FLIGHT DATA CHECK
    # --------------------------------------------------------

    validate_training_metadata()

    # --------------------------------------------------------
    # DEVICE
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
            ),
        )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_data_loaders()

    print(
        f"\nTraining samples      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples    : "
        f"{len(validation_dataset)}"
    )

    print(
        f"Batch size            : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Train batches/epoch   : "
        f"{len(train_loader)}"
    )

    # --------------------------------------------------------
    # LOAD V3 SOURCE MODEL
    # --------------------------------------------------------

    if not SOURCE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"V3 source model not found:\n"
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
        "Loaded V3 source checkpoint:"
    )

    print(
        SOURCE_MODEL_PATH
    )

    print(
        f"Source epoch          : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Source validation CER : "
        f"{checkpoint.get('validation_cer', float('nan')):.4f}"
    )

    print(
        f"Source validation WER : "
        f"{checkpoint.get('validation_wer', float('nan')):.4f}"
    )

    # --------------------------------------------------------
    # CHARSET
    # --------------------------------------------------------

    (
        index_to_char,
        blank_index,
    ) = load_charset()

    expected_num_classes = (
        max(
            index_to_char.keys()
        )
        + 1
    )

    expected_num_classes = max(
        expected_num_classes,
        blank_index + 1,
    )

    if num_classes != expected_num_classes:

        raise RuntimeError(
            "Checkpoint/charset class mismatch.\n"
            f"Checkpoint num_classes: {num_classes}\n"
            f"Charset requires      : "
            f"{expected_num_classes}"
        )

    print(
        f"Model classes         : "
        f"{num_classes}"
    )

    print(
        f"CTC blank index       : "
        f"{blank_index}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CTCLoss(
        blank=blank_index,
        reduction="mean",
        zero_infinity=True,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

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
    # V3 BASELINE ON CLEAN 23-LINE VALIDATION SET
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
    print("-" * 72)
    print("V3 BASELINE BEFORE V5 FINE-TUNING")
    print("-" * 72)

    print(
        f"Validation Loss : "
        f"{baseline_loss:.4f}"
    )

    print(
        f"Validation CER  : "
        f"{baseline_cer:.4f}"
    )

    print(
        f"Validation WER  : "
        f"{baseline_wer:.4f}"
    )

    print(
        f"Exact matches   : "
        f"{baseline_exact_matches}/"
        f"{len(validation_dataset)}"
    )

    print(
        f"Exact accuracy  : "
        f"{baseline_exact_accuracy * 100:.2f}%"
    )

    for number, (
        reference,
        prediction,
    ) in enumerate(
        baseline_examples,
        start=1,
    ):

        print()
        print(
            f"Baseline Example {number}:"
        )

        print(
            "GT  :",
            reference,
        )

        print(
            "PRED:",
            prediction,
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = {

        "version":
            "real_sinhala_ocr_v5",

        "configuration": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "initial_learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRAD_CLIP,

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

        "before_v5_finetuning": {

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

    # --------------------------------------------------------
    # BEST MODEL TRACKING
    #
    # Epoch 0 = unchanged V3 baseline.
    # This protects us if V5 training makes validation worse.
    # --------------------------------------------------------

    best_cer = (
        baseline_cer
    )

    best_wer = (
        baseline_wer
    )

    best_validation_loss = (
        baseline_loss
    )

    best_epoch = 0

    epochs_without_improvement = 0

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
        "Saved V3 baseline as V5 epoch-0 "
        "best checkpoint."
    )

    training_start = time.time()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print("=" * 72)

        print(
            f"V5 FINE-TUNE EPOCH "
            f"{epoch}/{EPOCHS}"
        )

        print("=" * 72)

        epoch_start = time.time()

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
        print("Epoch Summary")

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
            f"Validation WER  : "
            f"{validation_wer:.4f}"
        )

        print(
            f"Exact matches   : "
            f"{exact_matches}/"
            f"{len(validation_dataset)}"
        )

        print(
            f"Exact Accuracy  : "
            f"{exact_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch Time      : "
            f"{epoch_seconds:.1f}s"
        )

        # ----------------------------------------------------
        # EXAMPLES
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
                reference,
            )

            print(
                "PRED:",
                prediction,
            )

        # ----------------------------------------------------
        # SAVE LATEST CHECKPOINT
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

        # ----------------------------------------------------
        # BEST MODEL SELECTION
        #
        # 1. CER
        # 2. WER
        # 3. Validation loss
        # ----------------------------------------------------

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
                "BEST V5 CHECKPOINT UPDATED"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "No validation improvement."
            )

            print(
                "Early-stop counter: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

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

    history[
        "final"
    ] = {

        "baseline_cer":
            baseline_cer,

        "best_cer":
            best_cer,

        "baseline_wer":
            baseline_wer,

        "best_wer":
            best_wer,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_validation_loss,

        "training_time_seconds":
            total_training_time,
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

    print()
    print("=" * 72)
    print("REAL SINHALA OCR V5 FINE-TUNING COMPLETE")
    print("=" * 72)

    print(
        f"\nBaseline CER : "
        f"{baseline_cer:.4f}"
    )

    print(
        f"Best CER     : "
        f"{best_cer:.4f}"
    )

    print(
        f"Baseline WER : "
        f"{baseline_wer:.4f}"
    )

    print(
        f"Best WER     : "
        f"{best_wer:.4f}"
    )

    print(
        f"Best epoch   : "
        f"{best_epoch}"
    )

    print(
        f"Best loss    : "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Training time: "
        f"{total_training_time:.1f}s"
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
    print("IMPORTANT:")

    print(
        "The existing clean 23-line validation "
        "set was used only for development/model selection."
    )

    print(
        "Fresh unseen_003 through unseen_013 "
        "were NOT used during V5 training."
    )

    print(
        "A9 was NOT used during V5 training."
    )

    print("=" * 72)
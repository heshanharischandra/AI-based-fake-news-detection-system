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

AUGMENTATION_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation"
)

TRAIN_CSV = (
    AUGMENTATION_DIR
    / "real_finetune_train_augmented.csv"
)

VALIDATION_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits"
    / "real_finetune_validation.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


# Start again from the clean synthetic-trained model
SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_full"
    / "best_crnn_model.pt"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_augmented_finetuned"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_augmented_finetuned_model.pt"
)

LAST_MODEL_PATH = (
    OUTPUT_DIR
    / "last_augmented_finetuned_model.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "augmented_finetuning_history.json"
)


# ============================================================
# 2. TRAINING CONFIGURATION
# ============================================================

EPOCHS = 20

BATCH_SIZE = 4

# Lower learning rate because we already have a trained OCR model
LEARNING_RATE = 0.00005

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 5.0

NUM_WORKERS = 0

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 5


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
            f"Charset not found:\n"
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

    time_steps = log_probs.size(
        0
    )

    batch_size = log_probs.size(
        1
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
    device,
    epoch
):

    model.train()

    freeze_batchnorm_statistics(
        model
    )

    running_loss = 0.0

    valid_batches = 0

    total_batches = len(
        loader
    )

    start_time = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        images = (
            batch["images"]
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
                f"WARNING: invalid loss "
                f"at batch {batch_index}"
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

        if (
            batch_index == 1
            or batch_index % 20 == 0
            or batch_index == total_batches
        ):

            average_loss = (
                running_loss
                / valid_batches
            )

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Epoch {epoch} "
                f"| Batch "
                f"{batch_index}/"
                f"{total_batches} "
                f"| Loss "
                f"{average_loss:.4f} "
                f"| {elapsed:.1f}s"
            )

    if valid_batches == 0:

        return math.inf

    return (
        running_loss
        / valid_batches
    )


# ============================================================
# 9. VALIDATE
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

    total_word_errors = 0
    total_reference_words = 0

    exact_matches = 0

    sample_count = 0

    examples = []

    with torch.no_grad():

        for batch in loader:

            images = (
                batch["images"]
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

                sample_count += 1

                char_distance = (
                    levenshtein_distance(
                        list(reference),
                        list(prediction)
                    )
                )

                total_character_errors += (
                    char_distance
                )

                total_reference_characters += (
                    len(reference)
                )

                reference_words = (
                    reference.split()
                )

                prediction_words = (
                    prediction.split()
                )

                word_distance = (
                    levenshtein_distance(
                        reference_words,
                        prediction_words
                    )
                )

                total_word_errors += (
                    word_distance
                )

                total_reference_words += (
                    len(reference_words)
                )

                if reference == prediction:

                    exact_matches += 1

                if len(examples) < 2:

                    examples.append(
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
        examples
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
    validation_wer,
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

        "validation_wer":
            validation_wer,

        "num_classes":
            num_classes,

        "hidden_size":
            hidden_size,

        "batch_size":
            BATCH_SIZE,

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

        "training_dataset":
            str(
                TRAIN_CSV
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
        "AUGMENTED REAL-DOMAIN SINHALA OCR FINE-TUNING"
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
        f"\nCombined real train samples : "
        f"{len(train_dataset)}"
    )

    print(
        f"Real validation samples     : "
        f"{len(validation_dataset)}"
    )

    print(
        f"Batch size                  : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Train batches per epoch     : "
        f"{len(train_loader)}"
    )

    # --------------------------------------------------------
    # Load pretrained synthetic checkpoint
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
        "\nLoaded source model:"
    )

    print(
        SOURCE_MODEL_PATH
    )

    print(
        f"Source epoch           : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Source validation loss : "
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
    # CTC Loss
    # --------------------------------------------------------

    criterion = nn.CTCLoss(
        blank=blank_index,
        reduction="mean",
        zero_infinity=True
    )

    # --------------------------------------------------------
    # Optimizer
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
            patience=2,
            min_lr=1e-6
        )
    )

    # --------------------------------------------------------
    # Baseline validation
    # --------------------------------------------------------

    (
        baseline_loss,
        baseline_cer,
        baseline_wer,
        baseline_exact,
        baseline_examples
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
        "BEFORE AUGMENTED FINE-TUNING"
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

    print(
        f"Validation WER  : "
        f"{baseline_wer:.4f}"
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

            "initial_learning_rate":
                LEARNING_RATE,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

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
                )
        },

        "before_finetuning": {

            "validation_loss":
                baseline_loss,

            "validation_cer":
                baseline_cer,

            "validation_wer":
                baseline_wer
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

    best_wer = (
        baseline_wer
    )

    epochs_without_improvement = 0

    training_start = (
        time.time()
    )

    best_model_saved = False

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            "\n" + "-" * 72
        )

        print(
            f"AUGMENTED FINE-TUNE "
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "-" * 72
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
                epoch
            )
        )

        (
            validation_loss,
            validation_cer,
            validation_wer,
            exact_accuracy,
            examples
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

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        print(
            "\nEpoch Summary"
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
            f"Validation WER  : "
            f"{validation_wer:.4f}"
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
        # Show examples
        # ----------------------------------------------------

        for example_number, (
            reference,
            prediction
        ) in enumerate(
            examples,
            start=1
        ):

            print(
                f"\nExample "
                f"{example_number}:"
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
        # Save last checkpoint
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
            num_classes,
            hidden_size
        )

        # ----------------------------------------------------
        # Select best by CER
        # Tie-break using validation loss
        # ----------------------------------------------------

        improved = False

        if validation_cer < best_cer:

            improved = True

        elif (
            validation_cer == best_cer
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
                num_classes,
                hidden_size
            )

            best_model_saved = True

            print(
                "\nBest augmented "
                "fine-tuned checkpoint updated."
            )

        else:

            epochs_without_improvement += 1

            print(
                "\nNo CER improvement."
            )

            print(
                "Early-stop counter: "
                f"{epochs_without_improvement}/"
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

            "validation_wer":
                validation_wer,

            "exact_accuracy":
                exact_accuracy,

            "learning_rate":
                current_lr,

            "epoch_seconds":
                epoch_seconds
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
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEARLY STOPPING ACTIVATED"
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    total_seconds = (
        time.time()
        - training_start
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "AUGMENTED REAL FINE-TUNING COMPLETE"
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
        f"Before WER : "
        f"{baseline_wer:.4f}"
    )

    print(
        f"Best WER   : "
        f"{best_wer:.4f}"
    )

    print(
        f"Best validation loss : "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Total training time  : "
        f"{total_seconds:.1f}s"
    )

    if best_model_saved:

        print(
            "\nBest model:"
        )

        print(
            BEST_MODEL_PATH
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            "No augmented epoch improved "
            "over the starting model."
        )

    print(
        "\nHistory:"
    )

    print(
        HISTORY_PATH
    )
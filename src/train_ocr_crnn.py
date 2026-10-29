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

from ocr_crnn_model import (
    CRNN,
    get_num_classes,
    count_parameters
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

SPLIT_DIR = (
    OCR_ROOT
    / "metadata"
    / "splits"
)

TRAIN_CSV = SPLIT_DIR / "train.csv"

VALIDATION_CSV = (
    SPLIT_DIR
    / "validation.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_full"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_crnn_model.pt"
)

LAST_MODEL_PATH = (
    MODEL_DIR
    / "last_crnn_model.pt"
)

TRAINING_HISTORY_PATH = (
    MODEL_DIR
    / "training_history.json"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 15

BATCH_SIZE = 4

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 5.0

NUM_WORKERS = 0

RANDOM_SEED = 42

HIDDEN_SIZE = 256

EARLY_STOPPING_PATIENCE = 4


# ============================================================
# RANDOM SEED
# ============================================================

def set_random_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
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
# CTC INPUT LENGTHS
# ============================================================

def create_input_lengths(
    log_probs
):

    time_steps = log_probs.size(0)

    batch_size = log_probs.size(1)

    return torch.full(
        (batch_size,),
        time_steps,
        dtype=torch.long
    )


# ============================================================
# TRAIN ONE EPOCH
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
                f"WARNING: Invalid loss "
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
            or batch_index % 100 == 0
            or batch_index == total_batches
        ):

            elapsed = (
                time.time()
                - start_time
            )

            average_loss = (
                running_loss
                / valid_batches
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
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    valid_batches = 0

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
                continue

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
    num_classes
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

        "num_classes":
            num_classes,

        "hidden_size":
            HIDDEN_SIZE,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            optimizer.param_groups[0]["lr"],

        "image_height":
            48,

        "image_width":
            1024,

        "created_at":
            datetime.now().isoformat()
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "FULL SINHALA OCR CRNN + CTC TRAINING"
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
    # Dataset
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    ) = create_data_loaders()

    print(
        f"\nTrain samples      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_dataset)}"
    )

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    num_classes = (
        get_num_classes()
    )

    model = CRNN(
        num_classes=num_classes,
        hidden_size=HIDDEN_SIZE
    ).to(
        device
    )

    print(
        f"Output classes     : "
        f"{num_classes}"
    )

    print(
        f"Trainable params   : "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # CTC
    # --------------------------------------------------------

    criterion = nn.CTCLoss(
        blank=0,
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

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    )

    history = {

        "configuration": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "initial_learning_rate":
                LEARNING_RATE,

            "hidden_size":
                HIDDEN_SIZE,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE
        },

        "epochs": []
    }

    best_validation_loss = (
        math.inf
    )

    epochs_without_improvement = 0

    training_start = (
        time.time()
    )

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            "\n" + "-" * 72
        )

        print(
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

        validation_loss = (
            validate(
                model,
                validation_loader,
                criterion,
                device
            )
        )

        # Scheduler monitors validation loss
        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        print(
            f"\nEpoch {epoch} Summary"
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
            f"Learning Rate   : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch Time      : "
            f"{epoch_seconds:.1f}s"
        )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            LAST_MODEL_PATH,
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            validation_loss,
            num_classes
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

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
                num_classes
            )

            print(
                "Best checkpoint updated."
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement."
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

            "learning_rate":
                current_lr,

            "epoch_seconds":
                epoch_seconds
        })

        with open(
            TRAINING_HISTORY_PATH,
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
    # FINISH
    # ========================================================

    total_seconds = (
        time.time()
        - training_start
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "FULL TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nBest validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Total training time : "
        f"{total_seconds:.1f}s"
    )

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nTraining history:"
    )

    print(
        TRAINING_HISTORY_PATH
    )
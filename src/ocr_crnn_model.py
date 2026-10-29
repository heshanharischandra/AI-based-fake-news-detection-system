from pathlib import Path
import json

import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHARSET_JSON = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "metadata"
    / "charset.json"
)


# ============================================================
# LOAD NUMBER OF OUTPUT CLASSES
# ============================================================

def get_num_classes():

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

    return int(
        charset_data[
            "model_output_classes"
        ]
    )


# ============================================================
# BIDIRECTIONAL LSTM BLOCK
# ============================================================

class BidirectionalLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.linear = nn.Linear(
            hidden_size * 2,
            output_size
        )


    def forward(
        self,
        x
    ):

        recurrent, _ = self.lstm(
            x
        )

        recurrent = self.dropout(
            recurrent
        )

        time_steps, batch_size, features = (
            recurrent.size()
        )

        output = self.linear(
            recurrent.reshape(
                time_steps * batch_size,
                features
            )
        )

        output = output.reshape(
            time_steps,
            batch_size,
            -1
        )

        return output


# ============================================================
# CRNN MODEL
# ============================================================

class CRNN(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_size=256
    ):

        super().__init__()


        # ====================================================
        # CNN FEATURE EXTRACTOR
        #
        # Input:
        # [B, 1, 48, 1024]
        #
        # Output approximately:
        # [B, 512, 1, 256]
        # ====================================================

        self.cnn = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # 48 x 1024
            # → 24 x 512
            # ------------------------------------------------

            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            # ------------------------------------------------
            # Block 2
            # 24 x 512
            # → 12 x 256
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            # Reduce height only
            # 12 x 256
            # → 6 x 256

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),


            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            # 6 x 256
            # → 3 x 256

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),


            # ------------------------------------------------
            # Final CNN layer
            #
            # Height:
            # 3 → 1
            # ------------------------------------------------

            nn.Conv2d(
                512,
                512,
                kernel_size=(3, 3),
                stride=1,
                padding=(0, 1)
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            )
        )


        # ====================================================
        # SEQUENCE MODEL
        # ====================================================

        self.rnn = nn.Sequential(

            BidirectionalLSTM(
                input_size=512,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=0.20
            ),

            BidirectionalLSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=num_classes,
                dropout=0.20
            )
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        images
    ):

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        features = self.cnn(
            images
        )

        # Expected:
        # [B, 512, 1, W]

        batch_size, channels, height, width = (
            features.size()
        )


        if height != 1:

            raise RuntimeError(
                f"CRNN CNN output height must be 1, "
                f"but received {height}."
            )


        # Remove height dimension
        #
        # [B, 512, 1, W]
        # →
        # [B, 512, W]

        features = features.squeeze(
            2
        )


        # Convert width into sequence/time dimension
        #
        # [B, 512, W]
        # →
        # [W, B, 512]

        features = features.permute(
            2,
            0,
            1
        )


        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        logits = self.rnn(
            features
        )


        # ----------------------------------------------------
        # CTC expects log probabilities
        #
        # [Time, Batch, Classes]
        # ----------------------------------------------------

        log_probs = torch.log_softmax(
            logits,
            dim=2
        )


        return log_probs


# ============================================================
# MODEL INFORMATION
# ============================================================

def count_parameters(
    model
):

    return sum(
        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "CRNN + BiLSTM + CTC MODEL TEST"
    )

    print(
        "=" * 70
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
    # Number of classes
    # --------------------------------------------------------

    num_classes = (
        get_num_classes()
    )


    print(
        f"Output classes: "
        f"{num_classes}"
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CRNN(
        num_classes=num_classes,
        hidden_size=256
    ).to(
        device
    )


    total_parameters = (
        count_parameters(
            model
        )
    )


    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )


    # --------------------------------------------------------
    # Dummy batch
    #
    # Batch size 2
    # --------------------------------------------------------

    dummy_images = torch.randn(
        2,
        1,
        48,
        1024,
        device=device
    )


    print(
        "\nDummy input shape:"
    )

    print(
        dummy_images.shape
    )


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        output = model(
            dummy_images
        )


    print(
        "\nModel output shape:"
    )

    print(
        output.shape
    )


    # --------------------------------------------------------
    # CTC information
    # --------------------------------------------------------

    time_steps = (
        output.size(
            0
        )
    )

    batch_size = (
        output.size(
            1
        )
    )

    output_classes = (
        output.size(
            2
        )
    )


    print(
        f"\nCTC time steps : "
        f"{time_steps}"
    )

    print(
        f"Batch size     : "
        f"{batch_size}"
    )

    print(
        f"Classes        : "
        f"{output_classes}"
    )


    # --------------------------------------------------------
    # Check maximum target length
    # --------------------------------------------------------

    max_target_length = 75


    print(
        f"\nMax dataset target length: "
        f"{max_target_length}"
    )


    if time_steps >= max_target_length:

        print(
            "CTC sequence length check: PASSED"
        )

    else:

        print(
            "CTC sequence length check: FAILED"
        )


    print(
        "\nMODEL FORWARD PASS: PASSED"
    )
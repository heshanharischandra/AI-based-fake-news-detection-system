
# ============================================================
# CROSS-PLATFORM IDENTITY PREDICTION API
# Experimental Logistic Regression Prototype
# ============================================================

from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Reuse the already tested prediction functions.
from predict_identity import (
    load_identity_model,
    predict_identity,
    seen_in_training,
)


# ============================================================
# STEP 1 - CREATE API ROUTER
# ============================================================

router = APIRouter(
    tags=["Cross-Platform Identity Matching"]
)


# ============================================================
# STEP 2 - REQUEST DATA MODEL
# ============================================================

class IdentityPredictionRequest(BaseModel):

    publisher_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    website_url: str = Field(
        ...,
        min_length=8,
        max_length=2048,
    )

    account_url: str = Field(
        ...,
        min_length=8,
        max_length=2048,
    )


# ============================================================
# STEP 3 - LOAD TRAINED MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_trained_model():

    return load_identity_model()


# ============================================================
# STEP 4 - CREATE PREDICTION ENDPOINT
# ============================================================

@router.post("/predict-identity")
def predict_identity_endpoint(
    request: IdentityPredictionRequest
):

    publisher_name = (
        request.publisher_name.strip()
    )

    website_url = (
        request.website_url.strip()
    )

    account_url = (
        request.account_url.strip()
    )

    if not publisher_name:
        raise HTTPException(
            status_code=422,
            detail="Publisher name cannot be empty.",
        )

    # ----------------------------------------
    # LOAD EXISTING TRAINED MODEL
    # ----------------------------------------

    try:

        (
            model,
            feature_columns,
            reverse_labels,
            prototype_only,
        ) = get_trained_model()

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:

        raise HTTPException(
            status_code=503,
            detail=(
                "Identity model is unavailable: "
                + str(error)
            ),
        ) from error

    # ----------------------------------------
    # GENERATE EXPERIMENTAL PREDICTION
    # ----------------------------------------

    try:

        (
            platform,
            prediction,
            features,
        ) = predict_identity(
            model=model,
            feature_columns=feature_columns,
            reverse_labels=reverse_labels,
            publisher_name=publisher_name,
            website_url=website_url,
            account_url=account_url,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    # ----------------------------------------
    # CHECK TRAINING DATA OVERLAP
    # ----------------------------------------

    training_overlap = seen_in_training(
        publisher_name,
        account_url,
    )

    # ----------------------------------------
    # RETURN JSON RESULT
    # ----------------------------------------

    return {

        "status": "success",

        "component": (
            "Cross-Platform Source "
            "Identity Matching"
        ),

        "model_name": (
            "Logistic Regression Pilot"
        ),

        "prototype_only": prototype_only,

        "input": {

            "publisher_name": publisher_name,

            "website_url": website_url,

            "platform": platform,

            "account_url": account_url,

        },

        "prediction": {

            "identity_label": prediction,

            "verification_status": (
                "not_independently_verified"
            ),

        },

        "feature_explanation": features,

        "training_data_check": {

            "exact_pair_seen_in_training": (
                training_overlap
            ),

            "note": (
                "A pair may share a publisher "
                "or account with training data "
                "even when the exact pair is unseen."
            ),

        },

        "limitations": {

            "independently_validated": False,

            "account_ownership_confirmed": False,

            "warning": (
                "Experimental name/handle-based "
                "ML prediction only. "
                "Not proof of account ownership "
                "or news credibility."
            ),

        },

    }

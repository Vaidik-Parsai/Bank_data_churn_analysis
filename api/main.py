"""FastAPI production service for Bank Customer Churn prediction."""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

# Automatically register project root and src directory into sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sklearn.pipeline import Pipeline

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskBandEnum,
)
from churn.config import (
    FEATURE_COLUMNS,
    METADATA_OUTPUT_PATH,
    MODEL_OUTPUT_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churn-api")


def assign_risk_band(probability: float) -> RiskBandEnum:
    """Classify customer churn probability into actionable risk tiers."""
    if probability < 0.30:
        return RiskBandEnum.LOW
    elif probability < 0.60:
        return RiskBandEnum.MEDIUM
    else:
        return RiskBandEnum.HIGH


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager loading artifacts once at application startup."""
    logger.info("Initializing Bank Churn Prediction Service...")

    if not MODEL_OUTPUT_PATH.is_file():
        logger.error(f"Fitted model artifact not found at {MODEL_OUTPUT_PATH.resolve()}")
        raise RuntimeError(f"Fitted model artifact missing: {MODEL_OUTPUT_PATH}")

    if not METADATA_OUTPUT_PATH.is_file():
        logger.error(f"Model metadata not found at {METADATA_OUTPUT_PATH.resolve()}")
        raise RuntimeError(f"Model metadata missing: {METADATA_OUTPUT_PATH}")

    # Load artifacts into application state
    logger.info(f"Loading pipeline from {MODEL_OUTPUT_PATH.name}...")
    pipeline: Pipeline = joblib.load(MODEL_OUTPUT_PATH)
    app.state.model = pipeline

    logger.info(f"Loading metadata from {METADATA_OUTPUT_PATH.name}...")
    with open(METADATA_OUTPUT_PATH, "r", encoding="utf-8") as f:
        metadata: Dict[str, Any] = json.load(f)
    app.state.metadata = metadata
    app.state.threshold = float(metadata.get("decision_threshold", 0.35))

    logger.info(
        f"Model successfully loaded. Active decision threshold: {app.state.threshold:.2f}"
    )

    yield

    logger.info("Shutting down Bank Churn Prediction Service.")


app = FastAPI(
    title="Bank Customer Churn Prediction Service",
    description=(
        "Production-ready inference service for retail bank customer churn prediction. "
        "Built with Scikit-learn Pipeline and FastAPI. Optimizes ROC-AUC and delivers "
        "calibrated probability scores, binary decisions, and tiered risk bands."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Safe global exception handler preventing leakage of internal tracebacks."""
    logger.error(f"Unhandled error processing {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the prediction request.",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health and readiness probe",
)
async def health_check() -> HealthResponse:
    """Verify application liveness and confirm the model is loaded in memory."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    version = getattr(app.state, "metadata", {}).get("model_version", "1.0.0")
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=version,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Metadata"],
    summary="Model specification and performance metadata",
)
async def get_model_info() -> ModelInfoResponse:
    """Return model architecture details, cross-validation metrics, and decision rationale."""
    if not hasattr(app.state, "metadata"):
        raise HTTPException(status_code=503, detail="Model metadata not yet loaded")

    meta = app.state.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "Random Forest Pipeline"),
        model_version=meta.get("model_version", "1.0.0"),
        scikit_learn_version=meta.get("scikit_learn_version", "unknown"),
        primary_metric=meta.get("primary_metric", "roc_auc"),
        cross_validation_roc_auc_mean=meta.get("cross_validation_roc_auc_mean", 0.0),
        decision_threshold=meta.get("decision_threshold", 0.35),
        decision_threshold_rationale=meta.get("decision_threshold_rationale", ""),
        best_hyperparameters=meta.get("best_hyperparameters", {}),
        test_metrics_calibrated_threshold=meta.get("test_metrics_calibrated_threshold", {}),
        feature_importances=meta.get("features", {}).get("feature_importances", {}),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict churn probability for a single customer",
)
async def predict_single(customer: CustomerInput) -> PredictionResponse:
    """Generate churn probability, binary prediction, and risk band for one customer profile."""
    model: Pipeline = app.state.model
    threshold: float = app.state.threshold

    # Convert Pydantic model into single-row DataFrame matching training feature columns
    customer_dict = customer.model_dump()
    row_df = pd.DataFrame([customer_dict])[FEATURE_COLUMNS]

    try:
        prob = float(model.predict_proba(row_df)[0, 1])
    except Exception as e:
        logger.error(f"Inference computation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating prediction probabilities.",
        )

    prediction = bool(prob >= threshold)
    risk_band = assign_risk_band(prob)

    return PredictionResponse(
        customer_id=customer.CustomerId,
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        risk_band=risk_band,
        threshold_used=threshold,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Predict churn for a batch of customer profiles",
)
async def predict_batch(batch: BatchPredictionRequest) -> BatchPredictionResponse:
    """Perform vectorized batch scoring across up to 1,000 customer records."""
    model: Pipeline = app.state.model
    threshold: float = app.state.threshold

    records = [c.model_dump() for c in batch.customers]
    df = pd.DataFrame(records)[FEATURE_COLUMNS]

    try:
        probabilities = model.predict_proba(df)[:, 1]
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating batch prediction probabilities.",
        )

    responses: List[PredictionResponse] = []
    churn_count = 0

    for customer, prob_float in zip(batch.customers, probabilities):
        prob = float(prob_float)
        prediction = bool(prob >= threshold)
        if prediction:
            churn_count += 1
        responses.append(
            PredictionResponse(
                customer_id=customer.CustomerId,
                churn_probability=round(prob, 4),
                churn_prediction=prediction,
                risk_band=assign_risk_band(prob),
                threshold_used=threshold,
            )
        )

    total = len(batch.customers)
    churn_rate = round(churn_count / total, 4) if total > 0 else 0.0

    return BatchPredictionResponse(
        total_customers=total,
        churn_predicted_count=churn_count,
        churn_rate_predicted=churn_rate,
        predictions=responses,
    )


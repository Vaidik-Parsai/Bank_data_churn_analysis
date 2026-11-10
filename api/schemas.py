"""Pydantic request and response schemas for the FastAPI churn prediction service."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GeographyEnum(str, Enum):
    """Supported customer residency countries."""

    FRANCE = "France"
    GERMANY = "Germany"
    SPAIN = "Spain"


class GenderEnum(str, Enum):
    """Customer gender identity."""

    MALE = "Male"
    FEMALE = "Female"


class RiskBandEnum(str, Enum):
    """Customer attrition risk tiers."""

    LOW = "Low"          # Probability < 0.30
    MEDIUM = "Medium"    # 0.30 <= Probability < 0.60
    HIGH = "High"        # Probability >= 0.60


class CustomerInput(BaseModel):
    """Schema for a single customer profile."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CustomerId": 15634602,
                "CreditScore": 650,
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 4,
                "Balance": 85000.50,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 72500.00,
            }
        }
    )

    CustomerId: Optional[int] = Field(
        default=None,
        description="Optional customer account identifier for auditing",
        examples=[15634602],
    )
    CreditScore: int = Field(
        ...,
        ge=300,
        le=850,
        description="FICO credit score ranging from 300 to 850",
        examples=[650],
    )
    Geography: GeographyEnum = Field(
        ...,
        description="Country of customer residence",
        examples=[GeographyEnum.GERMANY],
    )
    Gender: GenderEnum = Field(
        ...,
        description="Customer gender",
        examples=[GenderEnum.FEMALE],
    )
    Age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Customer age in years (must be 18 to 100)",
        examples=[42],
    )
    Tenure: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of years customer has maintained an account (0 to 10)",
        examples=[4],
    )
    Balance: float = Field(
        ...,
        ge=0.0,
        description="Total deposit balance in customer accounts",
        examples=[85000.50],
    )
    NumOfProducts: int = Field(
        ...,
        ge=1,
        le=4,
        description="Number of banking products held (1 to 4)",
        examples=[2],
    )
    HasCrCard: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether the customer holds a credit card (1 = Yes, 0 = No)",
        examples=[1],
    )
    IsActiveMember: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether the customer actively engages with services (1 = Active, 0 = Inactive)",
        examples=[0],
    )
    EstimatedSalary: float = Field(
        ...,
        ge=0.0,
        description="Estimated annual customer income",
        examples=[72500.00],
    )


class PredictionResponse(BaseModel):
    """Schema for individual customer churn prediction."""

    customer_id: Optional[int] = Field(
        default=None,
        description="Customer ID associated with this prediction",
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that customer will churn",
        examples=[0.6421],
    )
    churn_prediction: bool = Field(
        ...,
        description="Binary churn decision based on calibrated decision threshold",
        examples=[True],
    )
    risk_band: RiskBandEnum = Field(
        ...,
        description="Risk tier: Low (<0.30), Medium (0.30-0.60), High (>=0.60)",
        examples=[RiskBandEnum.HIGH],
    )
    threshold_used: float = Field(
        ...,
        description="The decision threshold applied to classify churn",
        examples=[0.35],
    )


class BatchPredictionRequest(BaseModel):
    """Schema for batch inference requests."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customers": [
                    {
                        "CustomerId": 15634602,
                        "CreditScore": 650,
                        "Geography": "Germany",
                        "Gender": "Female",
                        "Age": 42,
                        "Tenure": 4,
                        "Balance": 85000.50,
                        "NumOfProducts": 2,
                        "HasCrCard": 1,
                        "IsActiveMember": 0,
                        "EstimatedSalary": 72500.00,
                    },
                    {
                        "CustomerId": 15634603,
                        "CreditScore": 720,
                        "Geography": "France",
                        "Gender": "Male",
                        "Age": 28,
                        "Tenure": 6,
                        "Balance": 0.0,
                        "NumOfProducts": 1,
                        "HasCrCard": 1,
                        "IsActiveMember": 1,
                        "EstimatedSalary": 48000.00,
                    },
                ]
            }
        }
    )

    customers: List[CustomerInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of customer profiles to evaluate (up to 1,000)",
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction results."""

    total_customers: int = Field(
        ...,
        description="Number of customer profiles processed",
        examples=[2],
    )
    churn_predicted_count: int = Field(
        ...,
        description="Count of customers flagged as at-risk",
        examples=[1],
    )
    churn_rate_predicted: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of the batch predicted to churn",
        examples=[0.50],
    )
    predictions: List[PredictionResponse] = Field(
        ...,
        description="Individual prediction results",
    )


class HealthResponse(BaseModel):
    """Schema for service health check."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_version: str = Field(..., examples=["1.0.0"])


class ModelInfoResponse(BaseModel):
    """Schema detailing model architecture, training performance, and metadata."""

    model_name: str = Field(..., examples=["Bank Customer Churn Predictor (Random Forest Pipeline)"])
    model_version: str = Field(..., examples=["1.0.0"])
    scikit_learn_version: str = Field(..., examples=["1.9.0"])
    primary_metric: str = Field(..., examples=["roc_auc"])
    cross_validation_roc_auc_mean: float = Field(..., examples=[0.857])
    decision_threshold: float = Field(..., examples=[0.35])
    decision_threshold_rationale: str
    best_hyperparameters: Dict[str, Any]
    test_metrics_calibrated_threshold: Dict[str, Any]
    feature_importances: Dict[str, float]

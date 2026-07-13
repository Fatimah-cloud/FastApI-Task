from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


# -----------------------------
# Request Models
# -----------------------------

class CustomerRequest(BaseModel):

    tenure_months: int = Field(..., ge=0, le=72)
    monthly_charges: float = Field(..., ge=15, le=120)
    total_charges: float = Field(..., ge=0, le=10000)
    support_tickets: int = Field(..., ge=0, le=10)

    contract_type: str
    internet_service: str
    has_streaming: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 3,
                "monthly_charges": 95.5,
                "total_charges": 286.5,
                "support_tickets": 5,
                "contract_type": "month-to-month",
                "internet_service": "fiber",
                "has_streaming": True
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerRequest]


# -----------------------------
# Response Models
# -----------------------------

class PredictionResponse(BaseModel):
    id: str
    churn_probability: float
    label: str
    threshold: float
    model_version: str
    created_at: datetime


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    metrics: dict
    features: dict


class PredictionListResponse(BaseModel):
    data: List[PredictionResponse]
    pagination: dict
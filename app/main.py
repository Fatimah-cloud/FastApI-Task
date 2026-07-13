from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from app.predictor import Predictor
from app.schemas import (
    CustomerRequest,
    BatchPredictionRequest,
    PredictionResponse,
    PredictionListResponse,
    ModelInfoResponse,
)
from app.errors import register_exception_handlers


# --------------------------------------------------
# Global objects
# --------------------------------------------------

predictor = Predictor()

predictions = {}


# --------------------------------------------------
# Lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor.load()
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Failed to load model:", e)
        predictor.model = None

    yield


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)


# --------------------------------------------------
# Middleware
# --------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):

    request_id = str(uuid4())

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health", response_model=dict)
def health():

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    return {"status": "ok"}


# --------------------------------------------------
# Model Info
# --------------------------------------------------

@app.get(
    "/v1/model",
    response_model=ModelInfoResponse,
)
def get_model():

    if predictor.metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Metadata unavailable."
        )

    return {
        "model_name": predictor.metadata["model_name"],
        "version": predictor.metadata["version"],
        "metrics": predictor.metadata["metrics"],
        "features": predictor.metadata["features"],
    }


# --------------------------------------------------
# Predict One Customer
# --------------------------------------------------

@app.post(
    "/v1/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
def predict(customer: CustomerRequest, response: Response):

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable."
        )

    try:
        predictor.validate_customer(customer.model_dump())
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    result = predictor.predict(customer.model_dump())

    prediction_id = f"pred_{uuid4().hex[:10]}"

    prediction = {
        "id": prediction_id,
        "churn_probability": result["probability"],
        "label": result["label"],
        "threshold": predictor.metadata["threshold"],
        "model_version": predictor.metadata["version"],
        "created_at": datetime.now(timezone.utc),
    }

    predictions[prediction_id] = prediction

    response.headers["Location"] = f"/v1/predictions/{prediction_id}"

    return prediction
# --------------------------------------------------
# Get One Prediction
# --------------------------------------------------

@app.get(
    "/v1/predictions/{prediction_id}",
    response_model=PredictionResponse,
)
def get_prediction(prediction_id: str):

    if prediction_id not in predictions:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    return predictions[prediction_id]


# --------------------------------------------------
# List Predictions
# --------------------------------------------------

@app.get(
    "/v1/predictions",
    response_model=PredictionListResponse,
)
def list_predictions(
    limit: int = 20,
    offset: int = 0,
    label: str | None = None,
):

    items = list(predictions.values())

    if label:
        items = [
            item for item in items
            if item["label"] == label
        ]

    total = len(items)

    return {
        "data": items[offset:offset + limit],
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
        },
    }


# --------------------------------------------------
# Delete Prediction
# --------------------------------------------------

@app.delete(
    "/v1/predictions/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_prediction(prediction_id: str):

    predictions.pop(prediction_id, None)


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

@app.post(
    "/v1/predictions:batch",
    response_model=dict,
)
def batch_predictions(batch: BatchPredictionRequest):

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable."
        )

    if len(batch.customers) > 100:
        raise HTTPException(
            status_code=413,
            detail="Batch size exceeds 100 customers."
        )

    results = []

    for customer in batch.customers:

        try:
            predictor.validate_customer(customer.model_dump())
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=str(e)
            )

        prediction = predictor.predict(customer.model_dump())

        results.append(
            {
                "label": prediction["label"],
                "churn_probability": prediction["probability"],
            }
        )

    return {
        "count": len(results),
        "predictions": results,
    }
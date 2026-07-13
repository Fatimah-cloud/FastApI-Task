# Customer Churn Prediction API

## Overview

This project exposes a Machine Learning model through a RESTful API using FastAPI.

The API predicts whether a customer is likely to churn based on customer information such as tenure, monthly charges, contract type, internet service, and support tickets.

The trained Random Forest model is loaded once during application startup and serves predictions through versioned REST endpoints.

---


# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | /health | Health check |
| GET | /v1/model | Model information |
| POST | /v1/predictions | Predict one customer |
| POST | /v1/predictions:batch | Predict multiple customers |
| GET | /v1/predictions | List stored predictions |
| GET | /v1/predictions/{id} | Retrieve one prediction |
| DELETE | /v1/predictions/{id} | Delete a prediction |

---

# Example Prediction Request

```json
{
  "tenure_months": 3,
  "monthly_charges": 95.5,
  "total_charges": 286.5,
  "support_tickets": 5,
  "contract_type": "month-to-month",
  "internet_service": "fiber",
  "has_streaming": true
}
```

Example Response

```json
{
  "id": "pred_123456",
  "churn_probability": 0.8611,
  "label": "churned",
  "threshold": 0.5,
  "model_version": "1.0.0",
  "created_at": "2026-07-12T15:20:00Z"
}
---


import joblib
import json
import pandas as pd
from pathlib import Path

# Paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


class Predictor:
    def __init__(self):
        self.model = None
        self.metadata = None

    def load(self):
        """Load the ML model and metadata."""
        self.model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

    def validate_customer(self, customer: dict):
        """
        Validate categorical fields using metadata.json.
        """

        features = self.metadata["features"]

        valid_contracts = features["contract_type"]["values"]
        valid_services = features["internet_service"]["values"]

        if customer["contract_type"] not in valid_contracts:
            raise ValueError(
                f"Invalid contract_type. Allowed values: {valid_contracts}"
            )

        if customer["internet_service"] not in valid_services:
            raise ValueError(
                f"Invalid internet_service. Allowed values: {valid_services}"
            )

    def predict(self, customer: dict):
        """Predict one customer."""

        df = pd.DataFrame([customer])

        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]

        return {
            "prediction": int(prediction),
            "label": self.metadata["classes"][prediction],
            "probability": float(probabilities[prediction]),
        }
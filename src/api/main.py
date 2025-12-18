from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, RiskPrediction
import mlflow
import pandas as pd

app = FastAPI(title="Credit Risk API")

# Load best MLflow model
model_name = "Credit_Risk_Model"
model_version = 1  # or "latest"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(customer_id: int, features: CustomerFeatures):
    input_df = pd.DataFrame([features.dict()])
    risk_prob = model.predict_proba(input_df)[:, 1][0]
    return RiskPrediction(customer_id=customer_id, risk_probability=risk_prob)

from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    ProductCategory: str
    Amount: float
    total_amount: float
    avg_amount: float
    transaction_count: int
    # Add all other features your model needs

class RiskPrediction(BaseModel):
    customer_id: int
    risk_probability: float

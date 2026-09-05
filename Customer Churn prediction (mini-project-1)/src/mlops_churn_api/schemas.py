from pydantic import BaseModel

class ChurnInput(BaseModel):
    """Schema representing the customer data required for churn prediction."""
    gender: str
    SeniorCitizen: int
    partner: str
    Dependents: str
    tenure: int
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
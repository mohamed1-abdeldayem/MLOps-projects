from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    """Schema representing the customer data required for churn prediction."""

    gender: str = Field(..., min_length=1)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    partner: str = Field(..., min_length=1)
    Dependents: str = Field(..., min_length=1)
    tenure: int = Field(..., ge=0, le=72)
    MultipleLines: str = Field(..., min_length=1)
    InternetService: str = Field(..., min_length=1)
    OnlineSecurity: str = Field(..., min_length=1)
    OnlineBackup: str = Field(..., min_length=1)
    DeviceProtection: str = Field(..., min_length=1)
    TechSupport: str = Field(..., min_length=1)
    StreamingTV: str = Field(..., min_length=1)
    StreamingMovies: str = Field(..., min_length=1)
    Contract: str = Field(..., min_length=1)
    PaperlessBilling: str = Field(..., min_length=1)
    PaymentMethod: str = Field(..., min_length=1)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

import pytest

from src.mlops_churn_api import Inference
from src.mlops_churn_api.schemas import ChurnInput


@pytest.fixture
def data():
    """Create valid customer data for inference tests."""
    return ChurnInput(
        gender="Male",
        SeniorCitizen=0,
        partner="Yes",
        Dependents="No",
        tenure=12,
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="No",
        Contract="One year",
        PaperlessBilling="Yes",
        PaymentMethod="Mailed check",
        MonthlyCharges=70.5,
        TotalCharges=846.0,
    )


@pytest.fixture
def model():
    """Create an Inference instance for testing."""
    return Inference()


def test_preprocess_input(data, model):
    """Test that input preprocessing produces the expected model features."""
    inference = model
    df = inference.preprocess_input(data)

    print(df)
    print(df.shape)
    print(df.columns.tolist())

    assert list(df.columns) == inference.feature_names


def test_predict(data, model):
    """Test that the inference model returns a valid prediction and probability."""
    result = model.predict(data)

    label = result["prediction"]
    probabilities = result["probability"]

    assert label in [0, 1]
    assert 0 <= probabilities <= 1

import pytest
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV

from training import ModelTrainer

@pytest.fixture
def model():
    """Create a ModelTrainer instance for testing."""
    return ModelTrainer()

def test_data_load(model:ModelTrainer):
    """Test that the dataset is loaded correctly."""
    df = model.data_load()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Churn" in df.columns



def test_pre_processing(model):
    """Test that preprocessing returns valid features and target data."""
    x, y = model.pre_processing()

    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert "Churn" not in x.columns
    assert len(x) == len(y)
    assert x.shape[1] > 0

def test_split_data(model):
    """Test that the dataset is split into compatible training and test sets."""
    x_train, x_test, y_train, y_test = model.split_data()

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    assert len(x_train) > len(x_test)

    assert list(x_train.columns) == list(x_test.columns)

def test_create_model(model):
    """Test that the model training configuration returns a GridSearchCV instance."""
    grid = model.create_model()

    assert isinstance(grid, GridSearchCV)

def test_train_model(model):
    """Test that the model is trained and returns a fitted pipeline."""
    trained_model, x_test, y_test = model.train_model()

    assert isinstance(trained_model, GridSearchCV)
    assert hasattr(trained_model, "best_estimator_")
    assert hasattr(trained_model, "best_params_")

    assert len(x_test) == len(y_test)

def test_evaluate_model(model, caplog):
    """Test that model evaluation logs the expected performance metrics."""
    trained_model, x_test, y_test = model.train_model()

    model.evaluate_model(
        x_test,
        y_test,
        trained_model
    )

    assert "Model Evaluation" in caplog.text
    assert "Accuracy" in caplog.text
    assert "Precision" in caplog.text
    assert "Recall" in caplog.text
    assert "F1 Score" in caplog.text

def test_save_model(model, tmp_path):
    """Test that the trained model is successfully saved to disk."""
    trained_model, _, _ = model.train_model()

    model_path = tmp_path / "model.pkl"

    model.model_path = model_path

    model.save_model(trained_model)

    assert model_path.exists()
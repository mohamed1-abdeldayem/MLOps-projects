from .schemas import ChurnInput
from .config import settings
import onnxruntime as ort
import pandas as pd
from pathlib import Path
import json

class Inference:
    """Handle model loading, input preprocessing, and churn prediction."""
    def __init__(self):
        """Initialize the ONNX model and load the expected feature names."""
        self.model_onnx_path=settings.model_onnx_path
        self.BASE_DIR = Path(__file__).resolve().parents[2]
        with open(self.BASE_DIR / "models" / "feature_names.json") as f:
           self.feature_names = json.load(f)
        self.model=ort.InferenceSession(self.model_onnx_path)

    def preprocess_input(self, data: ChurnInput) -> pd.DataFrame:
        """Preprocess input data to match the model's expected features.

        Args:
            data: Validated customer data used for prediction.

        Returns:
            A DataFrame containing the encoded features in the same
            order as the features used during model training.
        """

        df = pd.DataFrame([data.model_dump()])

        df = pd.get_dummies(df, drop_first=True)

        df = df.reindex(columns=self.feature_names, fill_value=0)

        return df


    def predict(self, data: ChurnInput):
        """Predict customer churn using the ONNX model.

        Args:
            data: Validated customer data used for prediction.

        Returns:
            A dictionary containing the predicted class and its probability.
        """
        df = self.preprocess_input(data)

        input_name = self.model.get_inputs()[0].name

        result = self.model.run(
            None,
            {input_name:  df.to_numpy(dtype="float32")}
        )

        prediction = int(result[0][0])
        probabilities = result[1][0]

        return {
            "prediction": prediction,
            "probability": float(probabilities[prediction])
        }










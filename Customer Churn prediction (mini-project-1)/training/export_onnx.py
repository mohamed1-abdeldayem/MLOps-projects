from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
from mlops_churn_api import settings

class ExportToOnnx:
    """Export a trained scikit-learn model to ONNX format."""
    def __init__(self)->None:
        """Initialize model paths used for loading and exporting."""
        self.model_onnx_path = settings.model_onnx_path
        self.model_pickle_path = settings.model_pickle_path

    def load_pickle(self)->type:
        """Load the trained model from a pickle file.

        Returns:
            The trained scikit-learn model.
        """
        model=joblib.load(self.model_pickle_path)
        return model

    def export_onnx(self)->None:
        """Convert the trained model to ONNX and save it to disk."""
        model = self.load_pickle()
        initial_type = [
            ("input", FloatTensorType([None, model.n_features_in_]))
        ]

        onnx_model=convert_sklearn(
            model,
            initial_types=initial_type
        )

        with open(self.model_onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

if __name__ == "__main__":
    exporter = ExportToOnnx()
    exporter.export_onnx()




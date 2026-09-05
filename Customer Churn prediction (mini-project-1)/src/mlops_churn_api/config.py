from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    DATA_PATH: str
    MODEL_PICKLE_PATH: str
    MODEL_ONNX_PATH: str
    model_config=SettingsConfigDict(
        env_file=BASE_DIR/".env",
        env_file_encoding="utf-8"
    )

    @property
    def data_path(self) -> Path:
        """Return the absolute path to the training dataset."""
        return BASE_DIR / self.DATA_PATH

    @property
    def model_pickle_path(self) -> Path:
        """Return the absolute path to the serialized model."""
        return BASE_DIR / self.MODEL_PICKLE_PATH

    @property
    def model_onnx_path(self) -> Path:
        """Return the absolute path to the ONNX model."""
        return BASE_DIR / self.MODEL_ONNX_PATH

settings = Settings()
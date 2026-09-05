from fastapi import FastAPI,Request
from .schemas import ChurnInput
from .inference import Inference
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources during startup."""
    app.state.model = Inference()
    yield


app = FastAPI(
    title="Churn API",
    description="API for predicting customer churn",
    version="0.0.1",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/predict",
    summary="Predict customer churn",
    description="Predict whether a customer is likely to churn based on their information."
)
def predict(data: ChurnInput,request:Request ) -> dict[str, int | float]:
    """Predict whether a customer is likely to churn.

    Args:
        data: Customer information validated by Pydantic.
        request: FastAPI request object used to access the loaded model.

    Returns:
        A dictionary containing the predicted class and its probability.
    """
    return request.app.state.model.predict(data)

@app.get("/health")
def health_check(request:Request):
    """Check the health of the API and model availability.

    Args:
        request: FastAPI request object used to access the application state.

    Returns:
        A dictionary containing the API status and model loading status.
    """
    return {
        "status": "healthy",
        "model_loaded": request.app.state.model is not None
    }

@app.get("/")
def root():
    """Return a message confirming that the API is running.

    Returns:
        A dictionary containing the API status message.
    """
    return {
        "message": "Churn API is running"
    }


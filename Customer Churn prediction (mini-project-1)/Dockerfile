FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

COPY README.md ./

RUN pip install uv

COPY src ./src

COPY models ./models

RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "mlops_churn_api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
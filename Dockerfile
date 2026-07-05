FROM python:3.11-slim

WORKDIR /app

# System deps needed by scientific python wheels (kept minimal for a small image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY results/ ./results/

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "alpha_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

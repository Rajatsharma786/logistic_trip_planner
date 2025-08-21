
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY notebooks ./notebooks
COPY src ./src
COPY data ./data
COPY src/model_folder_v1 ./src/model_folder_v1

EXPOSE 8000


ENV ARTIFACT_DIR="src/model_folder_v1"


CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

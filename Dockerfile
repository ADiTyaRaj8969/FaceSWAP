# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder

WORKDIR /app
COPY frontend/package*.json frontend/
RUN cd frontend && npm ci --silent
COPY frontend/ frontend/
RUN mkdir -p static/react && cd frontend && npm run build


# Stage 2: Python runtime
FROM python:3.10-slim

# System libs needed by opencv, insightface, mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: torch CPU wheels (separate index — must be isolated from PyPI installs)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu

# Step 2: all other Python deps (PyPI only, no index conflict)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy React build from stage 1
COPY --from=frontend-builder /app/static/react ./static/react

# Copy app source
COPY . .

RUN mkdir -p models uploads/temp outputs/results

ENV PORT=7860
EXPOSE 7860

# Fix CRLF line endings (Windows → Linux) then make executable
RUN sed -i 's/\r$//' startup.sh && chmod +x startup.sh

CMD ["./startup.sh"]

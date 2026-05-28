# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app
COPY frontend/package*.json frontend/
RUN cd frontend && npm ci --silent
COPY frontend/ frontend/
RUN mkdir -p static/react
RUN cd frontend && npm run build
# Output is at /app/static/react  (vite outDir: ../static/react)


# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.10-slim

# System libs required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only for cloud deployment)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy React build from stage 1
COPY --from=frontend-builder /app/static/react ./static/react

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p models uploads/temp outputs/results

# HuggingFace Spaces listens on 7860
ENV PORT=7860
EXPOSE 7860

# Make startup script executable
RUN chmod +x startup.sh

CMD ["./startup.sh"]

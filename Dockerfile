# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder

WORKDIR /app
COPY frontend/package*.json frontend/
RUN cd frontend && npm ci --silent
COPY frontend/ frontend/
RUN mkdir -p static/react
RUN cd frontend && npm run build


# Stage 2: Python runtime
FROM python:3.10-slim

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

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --from=frontend-builder /app/static/react ./static/react

COPY . .

RUN mkdir -p models uploads/temp outputs/results

ENV PORT=7860
EXPOSE 7860

# Fix Windows CRLF line endings then make executable
RUN sed -i 's/\r$//' startup.sh && chmod +x startup.sh

CMD ["./startup.sh"]

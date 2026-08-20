FROM node:24-alpine AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    DATA_AGENT_CONFIG_ROOT=/app/runtime/config \
    DATA_AGENT_KNOWLEDGE_IMPORT_ROOT=/app/runtime/knowledge \
    DATA_AGENT_CHECKPOINT_PATH=/app/runtime/state/conversation.sqlite

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY databases/ ./databases/
COPY knowledge/ ./knowledge/
COPY src/ ./src/
COPY docker/ ./docker/
COPY --from=frontend-builder /build/frontend/dist/client ./frontend/dist/client

RUN mkdir -p /app/runtime

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=2)"

ENTRYPOINT ["python", "/app/docker/entrypoint.py"]

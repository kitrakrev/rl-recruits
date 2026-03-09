# Stage 1: Build
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app/env
COPY . .
RUN uv sync --frozen --no-editable

# Stage 2: Runtime
FROM ${BASE_IMAGE}
WORKDIR /app/env
COPY --from=builder /app/env /app/env

ENV LLM_MODE=stub
ENV ANTHROPIC_API_KEY=""

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

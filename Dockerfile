# ── Hugging Face Spaces Docker Image ──────────────────────────────────────
# Runs the Gradio dashboard (port 7860) and auto-starts the env server (:8000)
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY env/ env/
COPY server/ server/
COPY training/ training/
COPY ui/ ui/
COPY client.py models.py __init__.py ./

# Install project + UI extras
RUN uv pip install --system -e ".[ui]"

# HF Spaces expects port 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV LLM_MODE=stub
ENV ANTHROPIC_API_KEY=""

EXPOSE 7860

# Launch the Gradio dashboard (auto-starts the env server on :8000)
CMD ["python", "ui/dashboard.py", "--start_server", "--host", "0.0.0.0", "--port", "7860"]

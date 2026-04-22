FROM python:3.11-slim

# System libraries required by pyaudio (PortAudio) and webrtcvad
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first so this layer is cached on code-only changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY . .

# Vosk model is expected as a bind mount at runtime (see compose.yaml)
ENV VOSK_MODEL_PATH=/models/vosk-model-small-ja-0.22

CMD ["uv", "run", "python", "realtime_mode.py"]

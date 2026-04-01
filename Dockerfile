# Tensors API for Tengu PaaS
# ComfyUI runs as a separate container via the img addon
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install tensors with server dependencies
COPY pyproject.toml uv.lock README.md ./
COPY tensors/ ./tensors/
RUN uv pip install --system '.[server]'

# Copy entrypoint
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 5000

CMD ["/app/start.sh"]

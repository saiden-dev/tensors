# Tensors + ComfyUI for RTX 5090 / CUDA 12.8
# tensors API exposed, ComfyUI internal only
FROM madiator2011/better-comfyui:slim-5090

WORKDIR /workspace

# ComfyUI is pre-installed in base image at /workspace/ComfyUI
# Install ComfyUI Manager
RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

# Install tensors with server dependencies
COPY . /tmp/tensors
RUN pip install /tmp/tensors'[server]' && \
    rm -rf /tmp/tensors

# Configure tensors
RUN mkdir -p /root/.config/tensors && \
    cat > /root/.config/tensors/config.toml << 'EOF'
[paths]
models_dir = "/workspace/ComfyUI/models"
checkpoints = "/workspace/ComfyUI/models/checkpoints"
loras = "/workspace/ComfyUI/models/loras"
vae = "/workspace/ComfyUI/models/vae"
embeddings = "/workspace/ComfyUI/models/embeddings"

[comfyui]
url = "http://127.0.0.1:8188"
EOF

# Startup script: ComfyUI internal + tensors API exposed
RUN cat > /workspace/start.sh << 'EOF'
#!/bin/bash
# Start ComfyUI in background (internal only, localhost)
python /workspace/ComfyUI/main.py --listen 127.0.0.1 --port 8188 &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI..."
until curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; do
    sleep 1
done
echo "ComfyUI ready"

# Start tensors API (exposed)
exec tsr serve --host 0.0.0.0 --port 8080
EOF
RUN chmod +x /workspace/start.sh

# Only expose tensors API
EXPOSE 8080

CMD ["/workspace/start.sh"]

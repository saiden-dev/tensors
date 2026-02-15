# Tensors + ComfyUI for RTX 5090 / CUDA 12.8
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

# Configure tensors for ComfyUI paths
RUN mkdir -p /root/.config/tensors && \
    echo '[paths]' > /root/.config/tensors/config.toml && \
    echo 'models_dir = "/workspace/ComfyUI/models"' >> /root/.config/tensors/config.toml && \
    echo 'checkpoints = "/workspace/ComfyUI/models/checkpoints"' >> /root/.config/tensors/config.toml && \
    echo 'loras = "/workspace/ComfyUI/models/loras"' >> /root/.config/tensors/config.toml && \
    echo 'vae = "/workspace/ComfyUI/models/vae"' >> /root/.config/tensors/config.toml && \
    echo 'embeddings = "/workspace/ComfyUI/models/embeddings"' >> /root/.config/tensors/config.toml

# Expose ports: ComfyUI (8188), tensors API (8080)
EXPOSE 8188 8080

# Default: start ComfyUI (override with docker run command for tensors serve)
CMD ["python", "/workspace/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188"]

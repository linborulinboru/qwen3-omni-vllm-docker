FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 13.0, then vLLM + Qwen3-Omni extras
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

RUN pip install --no-cache-dir vllm

RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers" \
    accelerate qwen-omni-utils -U

RUN mkdir -p /opt/vllm
WORKDIR /opt/vllm

# Set Hugging Face caches to /models (the mounted volume)
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models

# Create data directory for multimodal files
RUN mkdir -p /data

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
  CMD curl -f http://localhost:8901/health || exit 1

COPY chat-template.jinja2 /opt/vllm/chat-template.jinja2

EXPOSE 8901

ENTRYPOINT ["vllm", "serve"]
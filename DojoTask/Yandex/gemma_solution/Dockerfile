FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        git+https://github.com/huggingface/transformers.git \
        accelerate \
        bitsandbytes \
        safetensors

WORKDIR /workspace

COPY solution.py /workspace/solution.py
COPY weights/ /workspace/weights/

CMD ["python", "/workspace/solution.py"]

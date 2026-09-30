FROM lmsysorg/sglang@sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2
ARG CODEX_VERSION=0.156.1
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm git curl jq swig build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @openai/codex@${CODEX_VERSION} \
    && python3 -m pip install --no-cache-dir 'smac==2.3.1' 'ConfigSpace==1.2.1'
COPY src /opt/inferencebench/src
COPY agents /opt/inferencebench/agents
COPY recipes/gb200_r1 /opt/inferencebench/recipes/gb200_r1
COPY tests /opt/inferencebench/tests
RUN mkdir -p /home/agent/task /opt/inference_eval \
    && ln -s /opt/inferencebench/src/eval/inference/bin /opt/inference_eval/bin
ENV PYTHONPATH=/opt/inferencebench PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS=1 SGLANG_SHARED_EXPERT_TP1=0 NCCL_NVLS_ENABLE=0 \
    TORCH_SYMM_MEM_DISABLE_MULTICAST=1
WORKDIR /opt/inferencebench
ENTRYPOINT ["python3", "/opt/inferencebench/recipes/gb200_r1/experiment.py"]

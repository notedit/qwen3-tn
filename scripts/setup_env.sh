#!/usr/bin/env bash
# 一次性环境搭建:uv venv + 依赖 + Qwen3-0.6B-Base 下载
set -euo pipefail
ROOT=/opt/dlami/nvme/leolxliu/qwen3-tn
cd "$ROOT"

echo "== [1/3] create venv =="
uv venv .venv --python 3.12 --seed

echo "== [2/3] install deps =="
uv pip install --python .venv/bin/python \
  torch transformers datasets accelerate \
  openai tenacity pytest cn2an jinja2 tensorboard \
  "huggingface_hub[cli]"

echo "== [3/3] download Qwen3-0.6B-Base =="
.venv/bin/hf download Qwen/Qwen3-0.6B-Base

.venv/bin/python - <<'EOF'
import torch, transformers
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
print("tokenizer loaded, vocab:", tok.vocab_size, "| eos:", tok.eos_token)
EOF
echo "SETUP_DONE"

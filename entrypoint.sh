#!/bin/bash
set -e
# If Wan2GP ckpts are not present (e.g. volume mounted empty), download once
KEY_CKPT="ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors"
if [ ! -f "$KEY_CKPT" ]; then
  echo "Wan2GP ckpts not found; downloading to current dir (mount point)..."
  python3 download_models.py
fi
exec "$@"

#!/bin/bash
set -e
# RunPod network volume (optional): https://docs.runpod.io/storage/network-volumes
# When attached to Serverless endpoint, it mounts at /runpod-volume. Use it for models so
# first worker downloads once and all workers/restarts reuse the same data.
if [ -d "/runpod-volume" ]; then
  export HF_HOME=/runpod-volume/hf_cache
  export HF_HUB_CACHE=/runpod-volume/hf_cache/hub
  export TRANSFORMERS_CACHE=/runpod-volume/hf_cache/transformers
  mkdir -p /runpod-volume/hf_cache /runpod-volume/ckpts
  # Use volume for ckpts (so first worker downloads once, rest reuse)
  if [ "$(readlink -f ckpts 2>/dev/null)" != "/runpod-volume/ckpts" ]; then
    rm -rf ckpts
    ln -sf /runpod-volume/ckpts ckpts
  fi
fi
# Ckpts: baked in at build, or on volume; only download if missing
KEY_CKPT="ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors"
if [ ! -f "$KEY_CKPT" ]; then
  echo "Wan2GP ckpts not found; downloading..."
  python3 download_models.py
fi
exec "$@"

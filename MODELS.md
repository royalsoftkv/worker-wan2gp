# Model files

Two options: **baked into the image** (no setup) or **RunPod network volume** (persistent, shared, faster cold starts).

## Option A: Baked into image (default)

Models are downloaded **during `docker build`** and stored in the image (~45 GB). No volumes required.

| Path in image | Contents | Size |
|---------------|----------|------|
| `/models/hf_cache` | RealVisXL, SDXL VAE | ~7 GB |
| `/Wan2GP/ckpts/` | Wan2GP video models | ~35 GB |

```bash
docker build -t worker-runpod .
```

## Option B: RunPod network volume (recommended for Serverless)

RunPod [network volumes](https://docs.runpod.io/storage/network-volumes) mount at **`/runpod-volume`** on Serverless workers. When present, the worker uses them for models so the first worker downloads once and all others reuse the same data.

1. In RunPod console: **Storage → Create Network Volume** (e.g. 50 GB), same datacenter as your endpoint.
2. **Serverless → your endpoint → Edit → Advanced → Network Volumes** → attach the volume.
3. (Optional) Pre-populate via [S3-compatible API](https://docs.runpod.io/storage/s3-api) to avoid any download on first run.

The entrypoint uses:

- `/runpod-volume/hf_cache` — RealVisXL / Hugging Face cache  
- `/runpod-volume/ckpts` — Wan2GP checkpoints (symlinked from `/Wan2GP/ckpts`)

If the volume is empty, the first worker runs `download_models.py` and RealVis will download on first image job; later workers use the same data.

## Local runs with host mounts

```bash
mkdir -p ./models/ckpts ./models/hf_cache
docker run --rm --gpus all \
  -v "$(pwd)/models/ckpts:/Wan2GP/ckpts" \
  -v "$(pwd)/models/hf_cache:/models/hf_cache" \
  worker-runpod
```

Empty mounts trigger downloads on first run.

# Large model files and volume mounts

These paths hold large downloaded models. Mount them from the host so data lives outside the container and is reused across runs.

## 1. Wan2GP video models – `/Wan2GP/ckpts/`

**Source:** `download_models.py` (Hugging Face: DeepBeepMeep/Wan2.1, DeepBeepMeep/LTX_Video)

| Path / pattern | Description | Approx. size |
|----------------|-------------|--------------|
| `ckpts/ltxv_0.9.7_13B_*.safetensors` | Video transformer weights | ~25 GB each (2 variants) |
| `ckpts/T5_xxl_1.1/` | T5 text encoder | ~5 GB |
| `ckpts/ltxv_0.9.7_VAE.safetensors` | VAE | ~1 GB |
| `ckpts/ltxv_0.9.7_spatial_upscaler.safetensors` | Upscaler | ~1 GB |
| `ckpts/pose/`, `ckpts/depth/`, `ckpts/mask/`, `ckpts/wav2vec/` | Control / aux models | 100s MB–1 GB |
| `ckpts/flownet.pkl` | Flow net | ~100 MB |

**Total (one transformer variant):** ~35–40 GB.

**Host mount example:**
```bash
-v /path/on/host/ckpts:/Wan2GP/ckpts
```

---

## 2. Hugging Face cache (RealVisXL, VAE, CLIP) – `/data/hf_cache`

**Source:** `diffusers` and `transformers` (RealVisXL, SDXL VAE, CLIP for image generation and NSFW check)

- `SG161222/RealVisXL_V5.0` – ~6 GB  
- `madebyollin/sdxl-vae-fp16-fix` – ~300 MB  
- `openai/clip-vit-base-patch16` – ~600 MB  

**Total:** ~7 GB.

The image sets `HF_HOME=/data/hf_cache`, so all Hugging Face downloads go there.

**Host mount example:**
```bash
-v /path/on/host/hf_cache:/data/hf_cache
```

---

## Example: run with external model directories

```bash
# Create host dirs once
mkdir -p ./models/ckpts ./models/hf_cache

# Run with volumes (first run will download into these dirs)
docker run --rm --gpus all \
  -v "$(pwd)/models/ckpts:/Wan2GP/ckpts" \
  -v "$(pwd)/models/hf_cache:/data/hf_cache" \
  worker-runpod
```

For RunPod Serverless, configure the endpoint’s **Volume** to mount the same paths so all workers use the same model data.

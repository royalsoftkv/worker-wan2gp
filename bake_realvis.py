"""Pre-download RealVisXL and VAE into HF cache at Docker build time (no GPU)."""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TQDM"] = "1"

import torch
from diffusers import AutoencoderKL, AutoPipelineForText2Image

def main():
    cache = os.environ.get("HF_HOME", "/models/hf_cache")
    print(f"Baking RealVisXL into {cache}", flush=True)
    # Download only; no .to("cuda") so this runs during docker build
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipe = AutoPipelineForText2Image.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        vae=vae,
        use_safetensors=True,
        torch_dtype=torch.float16,
        variant="fp16",
        custom_pipeline="lpw_stable_diffusion_xl",
    )
    print("RealVisXL bake done", flush=True)

if __name__ == "__main__":
    main()

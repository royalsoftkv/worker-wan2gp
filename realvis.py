# RealVisXL image generation - load pipeline once, reuse for all jobs

from diffusers.models import AutoencoderKL
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch
import base64
import io
import time
import numpy as np
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from typing import Dict, Optional

# Patch: PyTorch 2.x + diffusers – scheduler must not call .numpy() on CUDA tensors
def _patch_dpm_scheduler():
    sched = DPMSolverMultistepScheduler
    _orig_set_timesteps = sched.set_timesteps
    def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
        for attr in ("lambda_t", "alphas_cumprod", "sigmas"):
            t = getattr(self, attr, None)
            if isinstance(t, torch.Tensor) and t.is_cuda:
                setattr(self, attr, t.cpu())
        return _orig_set_timesteps(self, num_inference_steps=num_inference_steps, device=device, **kwargs)
    sched.set_timesteps = set_timesteps
_patch_dpm_scheduler()

_pipe = None
MODEL_NAME = "SG161222/RealVisXL_V5.0"


def get_scheduler(scheduler_config: Dict, name: str):
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(scheduler_config),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True))()


def _load_pipeline(sampler: str):
    """Load pipeline once and cache in _pipe."""
    global _pipe
    if _pipe is not None:
        return _pipe
    print("Start realvis (loading model once)")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    _pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_NAME,
        vae=vae,
        use_safetensors=True,
        add_watermarker=False,
        torch_dtype=torch.float16,
        variant="fp16",
        custom_pipeline="lpw_stable_diffusion_xl",
    )
    _pipe.scheduler = get_scheduler(_pipe.scheduler.config, sampler)
    _pipe.to("cuda")
    print("RealVisXL pipeline loaded")
    return _pipe


def generate(params: Dict) -> Dict:
    """
    Generate one image. Returns dict with 'image' (base64), metadata, or 'error'.
    Pipeline is loaded once on first call and reused.
    """
    output = {}
    seed = params.get("seed", 1)
    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    width = params.get("width", 512)
    height = params.get("height", 512)
    num_inference_steps = params.get("num_inference_steps", 11)
    guidance_scale = params.get("guidance_scale", 7)
    sampler = params.get("sampler", "DPM++ 2M Karras")

    try:
        pipe = _load_pipeline(sampler)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {"error": str(e)}

    # Per-job scheduler (in case sampler differs from initial load)
    pipe.scheduler = get_scheduler(pipe.scheduler.config, sampler)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    time_start = time.time()
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    except Exception as e:
        print(f"Inference failed: {e}")
        return {"error": str(e)}

    print(f"RealVis inference: {time.time() - time_start:.1f}s")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    output["image"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
    output["prompt"] = prompt
    output["negative_prompt"] = negative_prompt
    output["width"] = width
    output["height"] = height
    output["num_inference_steps"] = num_inference_steps
    output["guidance_scale"] = guidance_scale
    output["sampler"] = sampler
    output["model_name"] = MODEL_NAME
    output["seed"] = seed
    return output

# RealVisXL image generation - same as docker/realvis.py
# Expects event["input"]["params"] and writes to output dict

from diffusers.models import AutoencoderKL
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
import torch
import base64
import io
import sys
import time
import subprocess
import os
import tempfile
import numpy as np
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from typing import Callable, Dict, Optional, Tuple

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

print("Start realvis")

params = event.get("input", {}).get("params", {})

seed = params.get("seed", 1)
model_name = 'SG161222/RealVisXL_V5.0'
prompt=params.get("prompt", "")
negative_prompt=params.get("negative_prompt", "")
width = params.get("width", 512)
height = params.get("height", 512)
num_inference_steps = params.get("num_inference_steps", 11)
guidance_scale = params.get("guidance_scale", 7)
sampler=params.get("sampler", 'DPM++ 2M Karras')

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
generator = torch.Generator()
generator.manual_seed(seed)

time_start = time.time()

def get_scheduler(scheduler_config: Dict, name: str) -> Optional[Callable]:
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(scheduler_config),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: None)()


try:
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
    )
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_name,
        vae=vae,
        use_safetensors=True,
        add_watermarker=False,
        torch_dtype=torch.float16,
        variant="fp16",
        custom_pipeline="lpw_stable_diffusion_xl"
    )
    pipe.scheduler = get_scheduler(pipe.scheduler.config, sampler)
    pipe.to("cuda")
except RuntimeError as e:
    print(f"Failed to load model: {e}")
    output["error"] = f"Failed to load model: {e}"

if "error" not in output:
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    output["image"] = base64.b64encode(image_bytes).decode("utf-8")
    output["prompt"] = prompt
    output["negative_prompt"] = negative_prompt
    output["width"] = width
    output["height"] = height
    output["num_inference_steps"] = num_inference_steps
    output["guidance_scale"] = guidance_scale
    output["sampler"] = sampler
    output["model_name"] = model_name
    output["seed"] = seed

print("End realvis")

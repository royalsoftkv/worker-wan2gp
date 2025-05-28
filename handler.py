"""Example handler file."""

import runpod
from PIL import Image
import base64
from io import BytesIO
from wgp import generate_video
import os
import traceback

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def dummy_send_cmd(event, payload=None):
    if(event != "preview"):
        print(f"[send_cmd] {event}: {payload}")


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]
    job_id = job["id"]

    prompt = job_input.get("prompt", "")
    base64_str = job_input.get("image", "")
    resolution = job_input.get("resolution", "832x1104")
    video_length = job_input.get("video_length", "33")
    num_inference_steps = job_input.get("num_inference_steps", 30)
    guidance_scale = job_input.get("guidance_scale", 5)
    flow_shift = job_input.get("flow_shift", 5)

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    task = {
        "id": job_id
    }
    params = {
        "prompt": prompt,
        "negative_prompt": "",
        "resolution": resolution,
        "video_length": video_length,
        "seed": -1,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "audio_guidance_scale": 5,
        "flow_shift": flow_shift,
        "embedded_guidance_scale": 6,
        "repeat_generation": 1,
        "multi_images_gen_type": 0,
        "tea_cache_setting": 0,
        "tea_cache_start_step_perc": 0,
        "loras_multipliers": '',
        "image_prompt_type": '5',
        "image_start": [image],
        "image_end": None,
        "model_mode": None,
        "video_source": None,
        "keep_frames_video_source": '',
        "video_prompt_type": '',
        "image_refs": None,
        "video_guide": None,
        "keep_frames_video_guide": '',
        "video_mask": None,
        "audio_guide": None,
        "sliding_window_size": 153,
        "sliding_window_overlap": 9,
        "sliding_window_overlap_noise": 20,
        "sliding_window_discard_last_frames": 0,
        "remove_background_images_ref": 1,
        "temporal_upsampling": '',
        "spatial_upsampling": '',
        "RIFLEx_setting": 0,
        "slg_switch": 0,
        "slg_layers": [9],
        "slg_start_perc": 10,
        "slg_end_perc": 90,
        "cfg_star_switch": 0,
        "cfg_zero_step": -1,
        "prompt_enhancer": '',
        "activated_loras": [],
        "state": {
            "model_filename": 'ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors',
            "advanced": False,
            "gen": {
                "file_list": [],
                "prompt_no" : 1
            },
            "loras": [],
            "loras_presets": [],
            "loras_names": [],
            "validate_success": 1,
        },
        "model_filename": 'ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors'
    }

    try:
        # Attempt to generate the video
        generate_video(task, dummy_send_cmd, **params)
    except Exception as e:
        return {
            "error": "Video generation failed",
            "details": str(e),
            "trace": traceback.format_exc()
        }

    video_path = f"/Wan2GP/outputs/{job_id}.mp4"

    if not os.path.exists(video_path):
        return {
            "error": "File not generated",
            "path": video_path
        }

    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        return {
            "error": "Failed to read video file",
            "details": str(e)
        }

    return {
        "video_base64": video_base64
    }


runpod.serverless.start({"handler": handler})

"""RunPod serverless handler - image (RealVisXL) + video (Wan2GP) generation.
Same input format as docker server.js /start endpoint."""

import runpod
from PIL import Image
import base64
from io import BytesIO
import os
import traceback
import random

# Video generation
from wgp import generate_video


def dummy_send_cmd(event, payload=None):
    if event != "preview":
        print(f"[send_cmd] {event}: {payload}")


def handle_image(job):
    """Image generation via RealVisXL; pipeline loaded once and reused."""
    import realvis
    job_input = job.get("input", {})

    params = {
        "prompt": job_input.get("prompt", ""),
        "negative_prompt": job_input.get("negative_prompt", ""),
        "width": job_input.get("width", 512),
        "height": job_input.get("height", 512),
        "num_inference_steps": job_input.get("num_inference_steps", 11),
        "guidance_scale": job_input.get("guidance_scale", 7),
        "sampler": job_input.get("sampler", "DPM++ 2M Karras"),
        "seed": job_input.get("seed") or random.randint(0, 2**32 - 1),
    }

    try:
        output = realvis.generate(params)
    except Exception as e:
        return {
            "error": "Image generation failed",
            "details": str(e),
            "trace": traceback.format_exc(),
        }

    if "error" in output:
        return {
            "error": output["error"],
            "details": output.get("details", ""),
        }

    return {
        "image_base64": output["image"],
        "prompt": output["prompt"],
        "negative_prompt": output["negative_prompt"],
        "width": output["width"],
        "height": output["height"],
        "num_inference_steps": output["num_inference_steps"],
        "guidance_scale": output["guidance_scale"],
        "sampler": output["sampler"],
        "model_name": output["model_name"],
        "seed": output["seed"],
    }


def handle_video(job):
    """Video generation via Wan2GP (same as video.py)."""
    job_input = job["input"]
    job_id = job.get("id", "local-test")

    base64_str = job_input.get("image", "")
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    resolution = job_input.get("resolution", "832x1104")
    video_length = job_input.get("video_length", 153)
    num_inference_steps = job_input.get("num_inference_steps", 10)
    guidance_scale = job_input.get("guidance_scale", 5)
    flow_shift = job_input.get("flow_shift", 5)

    try:
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]

        if not base64_str:
            raise ValueError("No image data provided.")

        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")
    except Exception as e:
        return {"error": "Invalid base64 image data", "details": str(e)}

    if prompt == "TEST":
        return f"Received, {prompt}!"

    task = {"id": job_id}
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
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
        "loras_multipliers": "",
        "image_prompt_type": "5",
        "image_start": [image],
        "image_end": None,
        "model_mode": None,
        "video_source": None,
        "keep_frames_video_source": "",
        "video_prompt_type": "",
        "image_refs": None,
        "video_guide": None,
        "keep_frames_video_guide": "",
        "video_mask": None,
        "audio_guide": None,
        "sliding_window_size": 153,
        "sliding_window_overlap": 9,
        "sliding_window_overlap_noise": 20,
        "sliding_window_discard_last_frames": 0,
        "remove_background_images_ref": 1,
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "RIFLEx_setting": 0,
        "slg_switch": 0,
        "slg_layers": [9],
        "slg_start_perc": 10,
        "slg_end_perc": 90,
        "cfg_star_switch": 0,
        "cfg_zero_step": -1,
        "prompt_enhancer": "",
        "activated_loras": [],
        "state": {
            "model_filename": "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors",
            "advanced": False,
            "gen": {"file_list": [], "file_settings_list": [], "prompt_no": 1},
            "loras": [],
            "loras_presets": [],
            "loras_names": [],
            "validate_success": 1,
        },
        "model_filename": "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors",
    }

    try:
        generate_video(task, dummy_send_cmd, **params)
    except Exception as e:
        return {
            "error": "Video generation failed",
            "details": str(e),
            "trace": traceback.format_exc(),
        }

    video_path = f"/Wan2GP/outputs/{job_id}.mp4"

    if not os.path.exists(video_path):
        return {"error": "File not generated", "path": video_path}

    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        return {"error": "Failed to read video file", "details": str(e)}

    return {"video_base64": video_base64}


def handler(job):
    """Route to image or video generation based on input (same as docker /start)."""
    job_input = job.get("input", {})

    # Video: has base64 image. Image: has prompt only.
    if job_input.get("image"):
        return handle_video(job)
    else:
        return handle_image(job)


def main():
    # Local test: run with test_input.json (image) or test_input_video.json (video) when not in RunPod
    use_video = os.environ.get("USE_VIDEO_TEST", "")
    test_file = "test_input_video.json" if (use_video and os.path.exists("test_input_video.json")) else "test_input.json"
    if os.path.exists(test_file) and (
        os.environ.get("LOCAL_TEST") or not os.environ.get("RUNPOD_SERVERLESS")
    ):
        import json
        with open(test_file, "r", encoding="utf-8") as f:
            job = json.load(f)
        if "id" not in job:
            job["id"] = "local-test"
        result = handler(job)
        # Save image/video to file (use -v ./output:/Wan2GP/output to get the file on host)
        output_dir = os.environ.get("OUTPUT_DIR", "/Wan2GP/output")
        if isinstance(result, dict):
            if "image_base64" in result:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, "test_output.png")
                data = base64.b64decode(result["image_base64"])
                with open(out_path, "wb") as f:
                    f.write(data)
                    f.flush()
                print(f"Saved image to {out_path} ({len(data)} bytes)")
                # Print result without the large base64 field
                result_short = {k: ("<base64...>" if k == "image_base64" else v) for k, v in result.items()}
                print(json.dumps(result_short, indent=2))
            elif "video_base64" in result:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, "test_output.mp4")
                data = base64.b64decode(result["video_base64"])
                with open(out_path, "wb") as f:
                    f.write(data)
                    f.flush()
                print(f"Saved video to {out_path} ({len(data)} bytes)")
                result_short = {k: ("<base64...>" if k == "video_base64" else v) for k, v in result.items()}
                print(json.dumps(result_short, indent=2))
            else:
                print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, indent=2))
    else:
        runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()

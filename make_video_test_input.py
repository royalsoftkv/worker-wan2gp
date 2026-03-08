#!/usr/bin/env python3
"""Build test_input_video.json from an image file for local video (image-to-video) testing."""
import base64
import json
import os
import sys

IMAGE_PATH = os.environ.get("VIDEO_TEST_IMAGE", "output/test_output.png")
OUT_PATH = "test_input_video.json"

def main():
    if not os.path.isfile(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}", file=sys.stderr)
        print("Generate an image first (run image test), or set VIDEO_TEST_IMAGE=/path/to/image.png", file=sys.stderr)
        sys.exit(1)
    with open(IMAGE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # Use smaller resolution/length for 12GB GPU; override with env VIDEO_RESOLUTION, VIDEO_LENGTH if needed
    resolution = os.environ.get("VIDEO_RESOLUTION", "512x512")
    video_length = int(os.environ.get("VIDEO_LENGTH", "81"))  # 81 frames fits ~12GB
    job = {
        "id": "local-test-video",
        "input": {
            "prompt": "woman dancing, smiling",
            "negative_prompt": "",
            "image": b64,
            "resolution": resolution,
            "video_length": video_length,
            "num_inference_steps": 10,
            "guidance_scale": 5,
            "flow_shift": 5,
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    print(f"Wrote {OUT_PATH} (image from {IMAGE_PATH}, {len(b64)} chars base64)")

if __name__ == "__main__":
    main()

# worker-runpod

RunPod serverless worker with same capabilities as the docker setup:
- **Image generation** (RealVisXL) - text-to-image
- **Video generation** (Wan2GP) - image-to-video

Same input format as the docker `/start` endpoint.

## Build

```bash
docker build -t worker-runpod .
```

## Test locally

From the project directory (with `test_input.json` present and no `RUNPOD_SERVERLESS` env set):

```bash
python handler.py
```

This runs one job using `test_input.json` as input and prints the result. For RunPod deployment, the container sets `RUNPOD_SERVERLESS`, so the worker starts normally instead of running the test payload.

## Model files

- **Default:** Models are baked into the image during build (~45 GB). No volumes needed.
- **RunPod (recommended):** Attach a [network volume](https://docs.runpod.io/storage/network-volumes); it mounts at `/runpod-volume`. The worker uses it for models so the first worker downloads once and rest reuse. See [MODELS.md](MODELS.md).

**Run locally:**
```bash
docker run --name worker-runpod --rm --gpus all \
  -v "$(pwd)/output:/Wan2GP/output" \
  worker-runpod
```

## Deploy to RunPod

1. Push image to a container registry.
2. Create a Serverless endpoint and point it to your image.
3. (Optional) Create a network volume, attach it to the endpoint — workers will use it for models at `/runpod-volume`.

## Input format

**Image generation** (no `image` in input):
```json
{
  "input": {
    "prompt": "...",
    "negative_prompt": "",
    "width": 512,
    "height": 512,
    "num_inference_steps": 11,
    "guidance_scale": 7,
    "sampler": "DPM++ 2M Karras",
    "seed": 12345
  }
}
```

**Video generation** (has `image` base64 in input):
```json
{
  "input": {
    "prompt": "...",
    "negative_prompt": "",
    "image": "<base64-encoded-image>",
    "resolution": "832x1104",
    "video_length": 153,
    "num_inference_steps": 10,
    "guidance_scale": 5,
    "flow_shift": 5
  }
}
```

### Test video locally (image-to-video)

1. Generate an image first (run the image test above); that saves `output/test_output.png`.
2. On the host, build the video test input from that image:
   ```bash
   python make_video_test_input.py
   ```
   (Uses `output/test_output.png` by default; or set `VIDEO_TEST_IMAGE=/path/to/image.png`.) This creates `test_input_video.json`.
3. Run the container with video test, mounting the video input and output:
   ```bash
   docker rm -f worker-runpod 2>/dev/null || true
   docker run --name worker-runpod --gpus all \
     -v "$(pwd)/output:/Wan2GP/output" \
     -v "$(pwd)/test_input_video.json:/Wan2GP/test_input_video.json" \
     -e USE_VIDEO_TEST=1 \
     worker-runpod
   ```
   Output video is written to `./output/test_output.mp4`.


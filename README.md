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

## Large model files (volume mounts)

The image does **not** bake in the large model downloads. Two paths should be mounted from the host so models are stored outside the container and reused:

| Container path       | Contents                    | See |
|----------------------|-----------------------------|-----|
| `/Wan2GP/ckpts`      | Wan2GP video models (~35 GB) | [MODELS.md](MODELS.md) |
| `/data/hf_cache`     | RealVisXL, VAE, CLIP (~7 GB) | [MODELS.md](MODELS.md) |

**Example – run with host directories and fixed container name (for `docker logs worker-runpod`):**
```bash
mkdir -p ./models/ckpts ./models/hf_cache ./output
docker rm -f worker-runpod 2>/dev/null || true
docker run --name worker-runpod --gpus all \
  -v "$(pwd)/models/ckpts:/Wan2GP/ckpts" \
  -v "$(pwd)/models/hf_cache:/data/hf_cache" \
  -v "$(pwd)/output:/Wan2GP/output" \
  worker-runpod
```
Test output is written to `./output/test_output.png` or `./output/test_output.mp4`. To re-run, remove the container first: `docker rm -f worker-runpod`.

On RunPod Serverless, attach a **Volume** to the endpoint and mount the same paths so all workers share the same model data.

## Deploy to RunPod

1. Push image to a container registry (Docker Hub, GHCR, etc.)
2. Create RunPod Serverless endpoint and attach a volume for `ckpts` and `hf_cache` (see [MODELS.md](MODELS.md)).
3. Point to your image

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
     -v "$(pwd)/models/ckpts:/Wan2GP/ckpts" \
     -v "$(pwd)/models/hf_cache:/data/hf_cache" \
     -v "$(pwd)/output:/Wan2GP/output" \
     -v "$(pwd)/test_input_video.json:/Wan2GP/test_input_video.json" \
     -e USE_VIDEO_TEST=1 \
     worker-runpod
   ```
   Output video is written to `./output/test_output.mp4`.

# RunPod serverless worker - image (RealVisXL) + video (Wan2GP) generation
# docker build -t worker-runpod .
# Deploy to RunPod Serverless

FROM runpod/base:0.6.3-cuda12.4.1

RUN apt update -y && apt install python-is-python3 git -y

# PyTorch (stable cu124 index; test index had missing nvidia-cudnn-cu12==9.1.0.70)
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Wan2GP
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /Wan2GP
RUN cd /Wan2GP && git checkout 86725a65d4e8fe5ba6addd9f847ec611039992a8
RUN python3 -m pip install -r /Wan2GP/requirements.txt
RUN python3 -m pip install sageattention==1.0.6 peft===0.17.0

# Image gen (RealVisXL) + RunPod
RUN python3 -m pip install diffusers transformers accelerate runpod~=1.7.9

ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV DISABLE_TQDM=1
# Hugging Face cache – all libs use this; mount a host dir to persist (e.g. -v ./models/hf_cache:/data/hf_cache)
ENV HF_HOME=/data/hf_cache
ENV HF_HUB_CACHE=/data/hf_cache/hub
ENV TRANSFORMERS_CACHE=/data/hf_cache/transformers

# Video models go to /Wan2GP/ckpts (mount a host dir; entrypoint downloads if empty)
ADD download_models.py /Wan2GP/
ADD entrypoint.sh /Wan2GP/
RUN chmod +x /Wan2GP/entrypoint.sh

# Worker files
ADD handler.py /Wan2GP/
ADD realvis.py /Wan2GP/
ADD wgp_config.json /Wan2GP/
ADD test_input.json /Wan2GP/
ADD make_video_test_input.py /Wan2GP/

# Patch wgp.py to use job_id for output filename
RUN sed -i "s|file_name = f\"{time_flag}_seed{seed}_{sanitize_file_name(save_prompt\\[:100\\]).strip()}.mp4\"|file_name = f\"{task['id']}.mp4\"|g" /Wan2GP/wgp.py

WORKDIR /Wan2GP
ENTRYPOINT ["/Wan2GP/entrypoint.sh"]
CMD ["python", "-u", "handler.py"]

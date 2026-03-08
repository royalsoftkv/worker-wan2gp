# RunPod serverless worker - image (RealVisXL) + video (Wan2GP) generation
# Models are baked into the image so RunPod can recreate containers without re-downloading.
# docker build -t worker-runpod .

FROM runpod/base:0.6.3-cuda12.4.1

RUN apt update -y && apt install python-is-python3 git -y

# PyTorch (stable cu124 index)
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
# Hugging Face cache – baked into image (no external volume needed on RunPod)
ENV HF_HOME=/models/hf_cache
ENV HF_HUB_CACHE=/models/hf_cache/hub
ENV TRANSFORMERS_CACHE=/models/hf_cache/transformers

# Bake RealVisXL + VAE into image (download at build time)
ADD bake_realvis.py /Wan2GP/
RUN mkdir -p /models/hf_cache && python3 /Wan2GP/bake_realvis.py

# Bake Wan2GP ckpts into image (~35 GB; long build)
ADD download_models.py /Wan2GP/
RUN cd /Wan2GP && python3 download_models.py

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

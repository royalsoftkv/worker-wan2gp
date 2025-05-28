FROM runpod/base:0.6.3-cuda12.4.1

RUN apt update -y && apt install python-is-python3 git -y

RUN python3 -m pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git
RUN python3 -m pip install -r /Wan2GP/requirements.txt
RUN python3 -m pip install sageattention==1.0.6
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ADD download_models.py /Wan2GP
RUN cd /Wan2GP && python download_models.py

#ADD test_generate_video.py /Wan2GP
ADD image.webp /Wan2GP

RUN python3 -m pip install runpod~=1.7.9
ADD handler.py /Wan2GP
#ADD generate_video.py /Wan2GP
ADD test_input.json /Wan2GP

RUN sed -i "s|file_name = f\"{time_flag}_seed{seed}_{sanitize_file_name(save_prompt\\[:100\\]).strip()}.mp4\"|file_name = f\"{task['id']}.mp4\"|g" Wan2GP/wgp.py


CMD cd /Wan2GP && python -u handler.py
#CMD bash
#CMD cd /Wan2GP && python test_generate_video.py

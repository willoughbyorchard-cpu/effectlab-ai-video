FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV COMFYUI_PATH=/opt/ComfyUI
ENV PYTHONPATH=/opt/ComfyUI

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git /opt/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-segment-anything-2.git /opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2

COPY requirements.txt /app-requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /opt/ComfyUI/requirements.txt
RUN if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt ]; then python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt; fi
RUN if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ]; then python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; fi
RUN if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess/requirements.txt ]; then python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess/requirements.txt; fi
RUN if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2/requirements.txt ]; then python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2/requirements.txt; fi
RUN python -m pip install -r /app-requirements.txt

COPY app.py /app/app.py
WORKDIR /app
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--timeout-keep-alive", "300"]

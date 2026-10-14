FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app-requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /app-requirements.txt

WORKDIR /app
COPY app.py /app/app.py
COPY scratch_model.py /app/scratch_model.py
COPY train_scratch.py /app/train_scratch.py
COPY prepare_data.py /app/prepare_data.py
COPY tests /app/tests
COPY static /app/static

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --timeout-keep-alive 300"]

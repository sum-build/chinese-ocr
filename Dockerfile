FROM python:3.9-slim-buster

ENV TZ=Asia/Shanghai

RUN pip3 install --upgrade --no-cache-dir \
&&  pip3 install --no-cache-dir tornado numpy opencv-python-headless onnxruntime Shapely pyclipper Pillow \
&&  apt update \
&&  apt upgrade -y \
&&  apt install -y libgeos-dev \
&&  apt clean \
&&  rm -rf /tmp/* && rm -rf /root/.cache/*

COPY . /chineseocr
WORKDIR /chineseocr

EXPOSE 80

CMD ["python3", "backend/main.py"]
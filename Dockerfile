FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# 기본 툴 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip vim cmake build-essential \
    python3.6 python3.6-dev python3-pip \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python3.6을 기본으로
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# TensorFlow GPU (TF1.x)
RUN pip install --no-cache-dir tensorflow-gpu==1.14.0

# MVSNet requirements (최적 버전)
RUN pip install --no-cache-dir \
    numpy==1.16.4 \
    scipy==1.2.2 \
    scikit-image==0.15.0 \
    imageio==2.6.1 \
    matplotlib==3.0.3 \
    opencv-python-headless==4.1.2.30 \
    Pillow==6.2.2 \
    tqdm==4.46.0

WORKDIR /workspace


ARG USERNAME=junho
ARG USER_UID=1004
ARG USER_GID=1005




# 유저 생성
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace

USER $USERNAME

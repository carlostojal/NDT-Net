FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# update system
RUN apt-get update && apt-get upgrade -y

# install build-essential
RUN apt install -y \
    build-essential \
    git \
    libgsl-dev \
    libx11-6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# install requirements
RUN pip install wandb open3d optuna numpy

# copy the code
COPY . /workspace
WORKDIR /workspace

# build ND-Net
WORKDIR /workspace/core_legacy/build
RUN cmake ..
RUN make -j8
RUN cp ./libndnet.so /usr/local/lib/libndnet.so

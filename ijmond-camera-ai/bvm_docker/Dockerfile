# Use NVIDIA's official CUDA image as the base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

COPY ./inference /app
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the default command for the container
ENTRYPOINT ["python3", "monitor.py"]

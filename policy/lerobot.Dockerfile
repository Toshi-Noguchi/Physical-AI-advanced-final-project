# LeRobot (official release) with SmolVLA: converts the raw demonstrations to a LeRobot v3.0 dataset and will
# serve the fine-tuned SmolVLA to the simulator. CUDA torch so the same image can run the policy on the GPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir "lerobot[dataset,smolvla]==0.6.1"
# The resolver picks the newest torchcodec, which is built for a newer torch; 0.8 matches torch 2.9.
RUN pip install --no-cache-dir "torchcodec==0.8.1"

WORKDIR /workspace

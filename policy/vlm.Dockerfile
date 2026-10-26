# Inspection VLM (Molmo2). Its remote code targets transformers 4.57 (per the model card), so it gets its own
# image rather than sharing LeRobot's newer transformers.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir "transformers==4.57.1" accelerate einops pillow "numpy>=2"

WORKDIR /workspace

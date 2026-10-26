"""Serve the inspection VLM (Molmo2) to the line's controller (scripts/run_cell.py).

    docker/vlm.sh policy/vlm_server.py                 # 127.0.0.1:6124, ~17 GB of GPU memory

inspect: the defect checklist on the station's top and side images (top view cropped to the box).
ask:     a yes/no question about any images, used for the controller's state checks (e.g. was the box picked).
"""

from __future__ import annotations

import argparse
import time

import torch
from protocol import VLM_ADDRESS, serve
from vlm_inspect import Molmo2Inspector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/Molmo2-8B")
    parser.add_argument("--top-crop", type=float, default=0.5)
    args = parser.parse_args()
    t0 = time.time()
    inspector = Molmo2Inspector(args.model)
    print(f"[molmo2] {args.model} loaded in {time.time() - t0:.0f}s ({torch.cuda.memory_allocated() / 2**30:.1f} GiB)",
          flush=True)
    serve({"inspect": lambda images: inspector.judge(images, "checklist", args.top_crop),
           "ask": lambda images, question: inspector.ask(images, question)}, VLM_ADDRESS, "molmo2")


if __name__ == "__main__":
    main()

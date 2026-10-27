"""Serve a fine-tuned SmolVLA to the simulator (grasp skill).

    docker/policy.sh outputs/models/smolvla_pick     # a pretrained_model directory written by lerobot-train

The simulator sends the overhead, side and wrist images (same resolutions as the training data), the 8-d state
and the task; the reply is the whole action chunk (absolute joint targets + suction command, 30 Hz).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from protocol import POLICY_ADDRESS, serve

CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")


class SmolVLA:
    def __init__(self, checkpoint: str):
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        self.device = torch.device("cuda")

    def act(self, images, task: str, state) -> dict:
        obs = {key: np.asarray(img, dtype=np.uint8) for key, img in zip(CAMERA_KEYS, images)}
        obs["observation.state"] = np.asarray(state, dtype=np.float32)
        batch = self.pre(prepare_observation_for_inference(obs, self.device, task))
        with torch.inference_mode():
            chunk = self.post(self.policy.predict_action_chunk(batch))
        return {"actions": chunk.reshape(-1, chunk.shape[-1]).float().cpu().numpy()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="a pretrained_model directory written by lerobot-train")
    args = parser.parse_args()
    t0 = time.time()
    model = SmolVLA(args.checkpoint)
    print(f"[smolvla] {args.checkpoint} loaded in {time.time() - t0:.0f}s "
          f"({torch.cuda.memory_allocated() / 2**30:.1f} GiB)", flush=True)
    serve({"act": model.act}, POLICY_ADDRESS, "smolvla")


if __name__ == "__main__":
    main()

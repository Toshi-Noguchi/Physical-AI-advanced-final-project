"""Capture an evaluation set for the inspection VLM: one box at a time stopped at the station, shot by the
overhead (1280x960) and side (960x720) inspection cameras as in operation: the arm pulled back out of the view
and the raking lights flashed on.

    docker/run.sh scripts/make_inspection_set.py --boxes 200 --out data/inspection/eval1

Writes <out>/NNNN_{overhead,side}.png and <out>/labels.jsonl (box spec with the ground-truth defects).
"""

import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--boxes", type=int, default=200)
parser.add_argument("--p-defect", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--out", default="data/inspection/eval1")
parser.add_argument("--no-raking", action="store_true", help="plain overhead lighting (for comparison)")
parser.add_argument("--arm-home", action="store_true", help="leave the arm at home instead of pulling it back")
parser.add_argument("--kinds", nargs="+", default=None, help="only these defect kinds (e.g. dent)")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import numpy as np  # noqa: E402
from box_textures import DEFECT_KINDS, sample_box_spec  # noqa: E402
from env import InspectionLineEnv  # noqa: E402
from PIL import Image  # noqa: E402
from skills import retract_for_inspection  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
env = InspectionLineEnv(app, out / "_env", seed=args.seed, cameras=("overhead", "side"), p_defect=args.p_defect)
env.set_belt(False)
env.step(30, auto_spawn=False)
if not args.arm_home:
    retract_for_inspection(env)
rng = np.random.default_rng(args.seed)

with open(out / "labels.jsonl", "w") as f:
    for i in range(args.boxes):
        for path in list(env.boxes):
            env.remove_box(path)
        env.randomize_lighting(rng, spread=0.2)
        spec = sample_box_spec(rng, f"eval_{i}", p_defect=args.p_defect, defect_kinds=tuple(args.kinds or DEFECT_KINDS))
        path = env.spawn_box(spec=spec, x=env.layout.inspect_x + rng.uniform(-0.04, 0.04))
        env.step(40, auto_spawn=False)
        shots = (env.capture("overhead"), env.capture("side")) if args.no_raking else env.capture_inspection()
        for cam, img in zip(("overhead", "side"), shots):
            Image.fromarray(img).save(out / f"{i:04d}_{cam}.png")
        spec = env.boxes[path]["spec"]
        f.write(json.dumps({"index": i, "defective": spec.is_defective, **spec.to_dict()}) + "\n")
        f.flush()
        if i % 20 == 0:
            print(f"[inspect-set] {i}/{args.boxes}", flush=True)
print(f"[inspect-set] done: {args.boxes} boxes -> {out}", flush=True)
app.close()

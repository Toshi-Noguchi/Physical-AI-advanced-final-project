"""Run the inspection line on its own (no policy yet): boxes flow, stop at the station, every camera is
captured, and the box is released downstream. Useful to eyeball the environment.

    docker/run.sh sim/run_env.py --seconds 40 --out outputs/env_check
    GUI=1 DISPLAY=:5 docker/run.sh sim/run_env.py --gui      # runs until the window is closed
"""

import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--seconds", type=float, default=None, help="sim seconds (default: 40 headless, forever in GUI)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--p-defect", type=float, default=0.6)
parser.add_argument("--out", default="outputs/env_check")
parser.add_argument("--video-every", type=int, default=0, help="save an overview frame every N steps (0 = off)")
parser.add_argument("--hold", type=float, default=1.5, help="seconds a box waits at the station")
args, _ = parser.parse_known_args()
if args.seconds is None:
    args.seconds = float("inf") if args.gui else 40.0

app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720})
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import InspectionLineEnv  # noqa: E402
from PIL import Image  # noqa: E402

out = Path(args.out)
(out / "inspect").mkdir(parents=True, exist_ok=True)
(out / "frames").mkdir(parents=True, exist_ok=True)

env = InspectionLineEnv(app, out, seed=args.seed, p_defect=args.p_defect)
if args.gui:
    from omni.kit.viewport.utility import get_active_viewport

    get_active_viewport().camera_path = "/World/Cameras/overview"
print(f"[run_env] layout: {env.layout}")
print(f"[run_env] robot dofs: {env.robot.dof_names}")
env.step(30, auto_spawn=False)  # let renderers warm up
env.spawn_box()
env.set_belt(True)

waiting_since, step, records = None, 0, []
while env.time < args.seconds and app.is_running():
    env.step()
    step += 1
    if env.waiting_box and waiting_since is None:
        waiting_since = env.time
    if env.waiting_box and env.time - waiting_since >= args.hold:
        spec = env.boxes[env.waiting_box]["spec"]
        for cam in ("overhead", "side", "wrist"):
            img = env.capture(cam)
            if img is not None:
                Image.fromarray(img).save(out / "inspect" / f"{spec.box_id}_{cam}.png")
        records.append(spec.to_dict())
        print(f"[run_env] t={env.time:5.1f}s inspected {spec.box_id} defective={spec.is_defective} "
              f"defects={[(m.kind, m.variant, m.face) for m in spec.defects]}")
        env.release_box()
        waiting_since = None
    if args.video_every and step % args.video_every == 0:
        img = env.capture("overview")
        if img is not None:
            Image.fromarray(img).save(out / "frames" / f"{step:06d}.png")

img = env.capture("overview")
if img is not None:
    Image.fromarray(img).save(out / "overview_last.png")
(out / "ground_truth.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
print(f"[run_env] done: {len(records)} boxes inspected, {len(env.dropped)} dropped")
app.close()

"""Evaluate the grasp VLA (SmolVLA server) in the cell, in the order the line runs: the box stops at the station
and is shot for inspection with the arm pulled back, the arm moves to its home pose, and the policy should then
pick the box with the suction cup and carry it back home, as in the demonstrations.

    docker/policy.sh outputs/models/smolvla_pick &
    docker/run.sh scripts/test_vla_pick.py --episodes 20 --out outputs/vla_pick

An episode succeeds when the box is held, lifted, and the arm is back near home.
"""

import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="pick up the cardboard box")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max-seconds", type=float, default=15.0, help="sim seconds per episode")
parser.add_argument("--exec-steps", type=int, default=25, help="actions executed from each chunk before re-planning")
parser.add_argument("--fps", type=int, default=30, help="control rate of the training data")
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--video", action="store_true", help="save a frame mosaic per control step")
parser.add_argument("--server-wait", type=float, default=600.0, help="seconds to wait for the SmolVLA server")
parser.add_argument("--out", default="outputs/vla_pick")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "policy")]

import numpy as np  # noqa: E402
from env import InspectionLineEnv  # noqa: E402
from PIL import Image  # noqa: E402
from protocol import ModelClient  # noqa: E402
from scene import ARM_HOME  # noqa: E402
from skills import arm_home, retract_for_inspection  # noqa: E402

VIEWS = {"overhead": (480, 640), "side": (480, 640), "wrist": (360, 640)}  # as recorded for training
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
env = InspectionLineEnv(app, out / "_env", seed=args.seed, cameras=(*VIEWS, "overview"), resolutions=VIEWS)
env.set_belt(False)
env.step(30, auto_spawn=False)
client = ModelClient(wait=args.server_wait)
rng = np.random.default_rng(args.seed)
substeps = int(round(1.0 / (args.fps * env.dt)))


def reset() -> str:
    env.release()
    for path in list(env.boxes):
        env.remove_box(path)
    env.set_base_pose(*env.layout.robot_home)
    env.set_arm(ARM_HOME)
    retract_for_inspection(env)
    path = env.spawn_box(x=env.layout.inspect_x + rng.uniform(-0.05, 0.05), max_yaw_deg=8.0)
    env.step(45, auto_spawn=False)
    env.capture_inspection()  # what the line does before deciding to pick
    arm_home(env)  # the policy's demonstrations all start from here
    return path


def mosaic() -> np.ndarray:
    tiles = [np.asarray(Image.fromarray(env.capture(n)).resize((427, 320))) for n in (*VIEWS, "overview")]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


results = []
for ep in range(args.episodes):
    path = reset()
    start_z = env.box_pose(path)[0][2]
    t_end, n_query, n_frame, max_grip, ok = env.time + args.max_seconds, 0, 0, 0.0, False
    ep_dir = out / f"ep{ep:02d}"
    if args.video:
        ep_dir.mkdir(exist_ok=True)
    while env.time < t_end and not ok:
        reply = client.act([env.capture(n) for n in VIEWS], args.task, env.droid_state())
        n_query += 1
        for a in reply["actions"][: args.exec_steps]:
            env.apply_droid_action(a)
            env.step(substeps, auto_spawn=False)
            max_grip = max(max_grip, float(a[7]))
            if args.video:
                Image.fromarray(mosaic()).save(ep_dir / f"{n_frame:05d}.jpg", quality=85)
                n_frame += 1
            lifted = env.box_pose(path)[0][2] - start_z
            if env.held_box() == path and lifted > 0.05 and np.abs(env.arm_q() - ARM_HOME).max() < 0.15:
                ok = True
                break
    lifted = float(env.box_pose(path)[0][2] - start_z)
    results.append({"episode": ep, "success": ok, "held": env.held_box() == path, "lift": round(lifted, 3),
                    "queries": n_query, "max_grip_cmd": round(max_grip, 3), "sim_time": round(env.time - t_end + args.max_seconds, 2)})
    print(f"[vla] ep{ep} {'SUCCESS' if ok else 'FAIL'} held={results[-1]['held']} lift={lifted:.3f} "
          f"queries={n_query} max_grip={max_grip:.2f} latency={reply['latency']:.2f}s", flush=True)

(out / "results.json").write_text(json.dumps(results, indent=2))
print(f"[vla] {sum(r['success'] for r in results)}/{len(results)} successful", flush=True)
client.close()
app.close()

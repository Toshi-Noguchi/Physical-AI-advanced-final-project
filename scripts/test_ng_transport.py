"""Check the rule-based NG handling end to end: for each box, pick it at the station (scripted expert), carry it
to the reject table, put it down, and drive back to the line. Saves overview frames for a video.

    docker/run.sh scripts/test_ng_transport.py --boxes 3 --out outputs/ng_transport
"""

import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--boxes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="outputs/ng_transport")
parser.add_argument("--frame-every", type=int, default=2, help="physics steps between video frames (2 = 30 fps)")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import numpy as np  # noqa: E402
from env import InspectionLineEnv  # noqa: E402
from PIL import Image  # noqa: E402
from skills import RejectTable, carry_to_table, place, return_to_line, scripted_pick  # noqa: E402

out = Path(args.out)
(out / "frames").mkdir(parents=True, exist_ok=True)
env = InspectionLineEnv(app, out, seed=args.seed, cameras=("overview",), p_defect=1.0)
env.set_belt(False)
env.step(30, auto_spawn=False)
rng = np.random.default_rng(args.seed)
table = RejectTable(env.layout)
steps = {"n": 0, "frame": 0}


def record(env_):
    steps["n"] += 1
    if steps["n"] % args.frame_every == 0:
        Image.fromarray(env_.capture("overview")).save(out / "frames" / f"{steps['frame']:06d}.jpg", quality=88)
        steps["frame"] += 1


results = []
for i in range(args.boxes):
    path = env.spawn_box(x=env.layout.inspect_x)
    env.step(45, auto_spawn=False)
    held = scripted_pick(env, path, rng, on_step=record)
    print(f"[ng] box {i}: picked={held}", flush=True)
    if not held:
        results.append({"box": path, "ok": False, "stage": "pick"})
        env.release()
        continue
    env.take_off_line(path)
    slot = carry_to_table(env, table, on_step=record)
    base_q = env.robot.get_dof_positions().numpy()[0, :3]
    print(f"[dbg] at table: slot {slot} target base {np.round(table.base_pose(slot), 3)} actual {base_q.round(3)} "
          f"link0 {env.arm_base_pose()[0].round(3)} cup {env.ee_pose()[0].round(3)}", flush=True)
    place(env, table, slot, path, on_step=record)
    print(f"[dbg] after place: box {env.box_pose(path)[0].round(3)} cup {env.ee_pose()[0].round(3)}", flush=True)
    return_to_line(env, table, slot, on_step=record)
    print(f"[dbg] back at line: base {env.robot.get_dof_positions().numpy()[0, :3].round(3)} home "
          f"{np.round(env.layout.robot_home, 3)} arm {env.arm_q().round(3)}", flush=True)
    env.step(30, auto_spawn=False)
    pos = env.box_pose(path)[0]
    size = env.boxes[path]["spec"].size
    cx, cy = env.layout.ng_table_center
    sx, sy, h = env.layout.ng_table_size
    on_table = abs(pos[0] - cx) < sx / 2 and abs(pos[1] - cy) < sy / 2 and pos[2] > h
    results.append({"box": path, "ok": bool(on_table), "slot": slot, "pos": pos.round(3).tolist(),
                    "size": list(size), "expected_z": round(table.stack_top[slot] - size[2] / 2, 3)})
    print(f"[ng] box {i}: slot {slot} at {pos.round(3)} on_table={on_table} t={env.time:.1f}s", flush=True)

(out / "results.json").write_text(json.dumps(results, indent=2))
print(f"[ng] {sum(r['ok'] for r in results)}/{len(results)} boxes on the reject table, {steps['frame']} frames", flush=True)
app.close()

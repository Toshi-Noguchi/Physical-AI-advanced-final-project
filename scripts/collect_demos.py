"""Collect scripted suction-grasp demonstrations (raw format; convert with policy/to_lerobot.py).

Physics runs at 60 Hz; the overhead, side and wrist cameras, the state and the action are recorded at 30 Hz.
Each episode starts near the home pose, picks the box at the station with the suction cup and carries it back
to the home pose. It re-randomises the box (size, look, defects, position, yaw), a queued box upstream, the
Ridgeback's pose, the arm's start pose and the lighting, then executes a fresh plan from sim/expert.py.
Only successful episodes (box still held at home) are kept.

    docker/run.sh scripts/collect_demos.py --out data/raw/run1 --worker 0 --chunk 0 --episodes 50

Layout: <out>/<episode>/{overhead,side,wrist}/NNNNNN.jpg, episode.npz (state, action, timestamp, phase, ...),
meta.json; failures go to <out>/failures.jsonl.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/raw/run1")
parser.add_argument("--worker", type=int, default=0)
parser.add_argument("--chunk", type=int, default=0)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=0, help="base seed; worker and chunk are mixed in")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--p-defect", type=float, default=0.5)
parser.add_argument("--p-queued", type=float, default=0.3, help="chance of a second box waiting upstream")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import numpy as np  # noqa: E402
import panda_kin as pk  # noqa: E402
from env import ARM_DOFS, InspectionLineEnv  # noqa: E402
from expert import PHASES, plan_pick  # noqa: E402
from grasp_target import box_in_link0, keep_out_spots  # noqa: E402
from PIL import Image  # noqa: E402
from scene import ARM_HOME, TOOL_TIP_Z  # noqa: E402

TASKS = ("pick up the cardboard box", "pick up the box", "lift the box off the conveyor",
         "grab the cardboard box with the suction cup")
# The VLA sees what the station already has plus the wrist: overhead and side are the 1280x960 / 960x720
# inspection views, rendered smaller (same field of view), in DROID's exterior_1 / exterior_2 / wrist slots.
CAMERAS = {"overhead": (480, 640), "side": (480, 640), "wrist": (360, 640)}

seed = args.seed * 1_000_000 + args.worker * 1_000 + args.chunk
rng = np.random.default_rng(seed)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
env = InspectionLineEnv(app, out / f"_env_w{args.worker:02d}", seed=seed, cameras=tuple(CAMERAS),
                        p_defect=args.p_defect, resolutions=CAMERAS)
env.set_belt(False)
env.step(30, auto_spawn=False)
lay = env.layout
q_home = np.array(ARM_HOME)
substeps = int(round(1.0 / (args.fps * env.dt)))
assert abs(substeps * env.dt * args.fps - 1.0) < 1e-6, "physics rate must be a multiple of the record rate"


def reset_scene() -> tuple[str, dict]:
    """Clear the line, place a new box (and maybe a queued one), and re-randomise robot and lights."""
    env.release()  # a box still hanging from the cup would be parked invisible right under it
    for path in list(env.boxes):
        env.remove_box(path)
    rx, ry, ryaw = lay.robot_home
    base = (rx + rng.uniform(-0.03, 0.03), ry + rng.uniform(-0.03, 0.03), ryaw + np.radians(rng.uniform(-3, 3)))
    env.set_base_pose(*base)
    q0 = np.clip(q_home + rng.normal(0, [0.06] * 6 + [0.15]), pk.Q_MIN, pk.Q_MAX)
    env.set_arm(q0)
    env.randomize_lighting(rng)
    path = env.spawn_box(x=lay.inspect_x + rng.uniform(-0.05, 0.05), max_yaw_deg=8.0)
    queued = None
    if rng.random() < args.p_queued:
        queued = env.spawn_box(x=lay.inspect_x - rng.uniform(0.6, 1.0))
    env.step(45, auto_spawn=False)  # boxes settle, the arm holds its start pose
    return path, {"base_pose": list(map(float, base)), "arm_start": q0.tolist(),
                  "queued_box": env.boxes[queued]["spec"].to_dict() if queued else None}


def run_episode(name: str) -> dict:
    path, meta = reset_scene()
    center, box_rot, spec = box_in_link0(env, path)
    box_pos = env.box_pose(path)[0]
    base_pos, base_rot = env.arm_base_pose()
    q_start = env.arm_q()
    reasons = []
    for _ in range(30):
        plan, why = plan_pick(rng, q_start, q_home, center, box_rot, spec.size, keep_out_spots(spec), TOOL_TIP_Z,
                              env.dt)
        if plan is not None:
            break
        reasons.append(why)
    if plan is None:
        return {"ok": False, "reason": f"no feasible plan: {sorted(set(reasons))[:3]}", "box": spec.to_dict(),
                "q_start": q_start.tolist(), "box_center_link0": center.tolist()}

    ep_dir = out / name
    for cam in CAMERAS:
        (ep_dir / cam).mkdir(parents=True, exist_ok=True)
    task = str(rng.choice(TASKS))
    rec = {k: [] for k in ("state", "action", "timestamp", "phase", "exec_target", "ee_pos", "box_pos", "base_q", "grip_status")}
    start_z, t0, n = box_pos[2], env.time, len(plan.q)
    gripped_at_lift = None
    for k in range(n):
        if k % substeps == 0:
            i = len(rec["timestamp"])
            for cam in CAMERAS:
                Image.fromarray(env.capture(cam)).save(ep_dir / cam / f"{i:06d}.jpg", quality=95)
            nxt = min(k + substeps - 1, n - 1)  # the plan point at the next recorded frame
            rec["state"].append(env.droid_state())
            rec["action"].append(np.r_[plan.q[nxt], plan.grip[nxt]].astype(np.float32))
            rec["timestamp"].append(env.time - t0)
            rec["phase"].append(plan.phase[k])
            rec["exec_target"].append(plan.q[k] + plan.noise[k])
            rec["ee_pos"].append(env.ee_pose()[0])
            rec["box_pos"].append(env.box_pose(path)[0])
            rec["base_q"].append(env.robot.get_dof_positions().numpy()[0, :3].copy())
            rec["grip_status"].append(int(env.gripper.get_surface_gripper_status()[0]))
        env.set_robot_targets(**dict(zip(ARM_DOFS, map(float, plan.q[k] + plan.noise[k]))))
        if (plan.grip[k] > 0.5) != env.suction_on:
            env.set_suction(bool(plan.grip[k] > 0.5))
        env.step(1, auto_spawn=False)
        if gripped_at_lift is None and plan.phase[k] == 2:
            gripped_at_lift = [str(o) for o in env.gripper.get_gripped_objects()[0]]

    lifted = float(env.box_pose(path)[0][2] - start_z)
    success = env.held_box() == path and lifted > 0.05  # carried back home (box ~8-16 cm above the belt)
    if not success:
        grip_at = int(np.argmax(np.asarray(rec["phase"]) == 1))
        contact_w = base_pos + base_rot @ np.asarray(plan.info["contact"])
        fail_dir = out / "_failed" / name
        fail_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fail_dir / "episode.npz", **{k: np.asarray(v) for k, v in rec.items()},
                            plan_q=plan.q, plan_noise=plan.noise)
        shutil.copytree(ep_dir, fail_dir, dirs_exist_ok=True)
        return {"ok": False, "reason": f"not lifted (gripped={gripped_at_lift}, target={path}, rise={lifted:.3f}, cup at grip "
                f"{np.round(rec['ee_pos'][grip_at], 3).tolist()} vs plan {np.round(contact_w, 3).tolist()}, "
                f"box top z {box_pos[2] + spec.size[2] / 2:.3f})", "box": spec.to_dict(), "plan": plan.info,
                "_cleanup": str(ep_dir)}
    np.savez_compressed(ep_dir / "episode.npz", **{k: np.asarray(v) for k, v in rec.items()})
    meta.update(task=task, fps=args.fps, frames=len(rec["timestamp"]), cameras={c: list(r) for c, r in CAMERAS.items()},
                box=spec.to_dict(), box_defective=spec.is_defective, plan=plan.info, lift_height=lifted,
                phases=list(PHASES), seed=seed)
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"ok": True, "frames": len(rec["timestamp"]), "lift": lifted}


t_start = time.time()
n_ok = 0
for e in range(args.episodes):
    name = f"w{args.worker:02d}_c{args.chunk:03d}_e{e:03d}"
    t_ep = time.time()
    res = run_episode(name)
    if "_cleanup" in res:
        shutil.rmtree(res.pop("_cleanup"))
    n_ok += res["ok"]
    if not res["ok"]:
        with open(out / "failures.jsonl", "a") as f:
            f.write(json.dumps({"episode": name, **res}) + "\n")
    print(f"[collect] {name} {'OK ' if res['ok'] else 'FAIL'} {res.get('frames', '')} "
          f"{res.get('reason', '')} ({time.time() - t_ep:.1f}s)", flush=True)
print(f"[collect] worker {args.worker} chunk {args.chunk}: {n_ok}/{args.episodes} kept in "
      f"{(time.time() - t_start) / 60:.1f} min", flush=True)
shutil.rmtree(out / f"_env_w{args.worker:02d}", ignore_errors=True)
app.close()

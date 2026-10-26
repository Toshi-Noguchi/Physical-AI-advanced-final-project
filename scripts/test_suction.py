"""Functional check of the suction gripper: servo above a box at the station, descend, grip, lift,
put it back down and release.

    docker/run.sh scripts/test_suction.py
"""

import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import numpy as np  # noqa: E402
from env import InspectionLineEnv  # noqa: E402
from PIL import Image  # noqa: E402

out = Path("outputs/test_suction")
out.mkdir(parents=True, exist_ok=True)
env = InspectionLineEnv(app, out, seed=3, p_defect=1.0, cameras=("overview", "wrist", "side"))
t0 = env.time
env.step(60, auto_spawn=False)
print(f"[suction] physics dt={env.dt:.4f}  60 steps -> {env.time - t0:.3f} s sim time", flush=True)
print(f"[suction] jacobian shape={env.robot.jacobian_matrix_shape}  ee={env.ee_pose()[0]}", flush=True)
print(f"[suction] base q={env.robot.get_dof_positions().numpy()[0][:3].round(4)} "
      f"target={env.robot.get_dof_position_targets().numpy()[0][:3].round(4)}", flush=True)


def snap(tag: str) -> None:
    env.step(3, auto_spawn=False)
    for cam in ("overview", "wrist", "side"):
        img = env.capture(cam)
        if img is not None:
            Image.fromarray(img).save(out / f"{tag}_{cam}.png")


path = env.spawn_box(x=env.layout.inspect_x)
spec = env.boxes[path]["spec"]
env.step(90, auto_spawn=False)
box, _ = env.box_pose(path)
top = box[2] + spec.size[2] / 2
print(f"[suction] box {spec.box_id} size={spec.size} centre={box.round(3)} top={top:.3f}", flush=True)
snap("0_start")

above = np.array([box[0], box[1], top + 0.12])
ok = env.move_tool(above, timeout=8.0)
print(f"[suction] above: reached={ok} ee={env.ee_pose()[0].round(3)}", flush=True)
snap("1_above")

ok = env.move_tool([box[0], box[1], top + 0.004], timeout=5.0, tol=0.002)
env.set_suction(True)
env.step(30, auto_spawn=False)
status = env.gripper.get_surface_gripper_status()
print(f"[suction] contact: reached={ok} ee={env.ee_pose()[0].round(3)} status={status} "
      f"gripped={env.gripper.get_gripped_objects()} held={env.held_box()}", flush=True)
snap("2_grip")

ok = env.move_tool(above + [0, 0, 0.08], timeout=6.0)
lifted, _ = env.box_pose(path)
print(f"[suction] lift: reached={ok} box z {box[2]:.3f} -> {lifted[2]:.3f} held={env.held_box()}", flush=True)
snap("3_lift")

env.move_tool([box[0], box[1], top + 0.01], timeout=6.0)
env.set_suction(False)
env.step(60, auto_spawn=False)
env.move_tool(above, timeout=5.0)
final, _ = env.box_pose(path)
print(f"[suction] release: box z={final[2]:.3f} held={env.held_box()}", flush=True)
snap("4_release")
print("[suction] RESULT", "PASS" if lifted[2] > box[2] + 0.15 and env.held_box() is None else "FAIL", flush=True)
app.close()

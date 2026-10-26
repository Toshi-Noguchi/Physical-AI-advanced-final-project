"""Save every camera's view at the home pose and just before the cup touches a box.

    docker/run.sh scripts/snap_views.py --out tmp
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="tmp")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--gap", type=float, default=0.015, help="cup-to-box-top distance for the pre-grasp shot")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from env import CAMERA_PATHS, InspectionLineEnv  # noqa: E402
from PIL import Image  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
env = InspectionLineEnv(app, out / "_env", seed=args.seed, p_defect=1.0, cameras=tuple(CAMERA_PATHS))


def snap(tag: str) -> None:
    env.step(10, auto_spawn=False)  # let every render product catch up with the final pose
    for name in env.cameras:
        img = env.capture(name)
        if img is not None:
            Image.fromarray(img).save(out / f"{tag}_{name}.png")
    print(f"[snap] {tag}: cup={env.ee_pose()[0].round(3)} arm q={env.arm_q().round(3)}", flush=True)


path = env.spawn_box(x=env.layout.inspect_x)
spec = env.boxes[path]["spec"]
env.step(90, auto_spawn=False)
env.go_home()
snap("0_home")

box, _ = env.box_pose(path)
top = box[2] + spec.size[2] / 2
tool_quat = env.ee_pose()[1]  # keep the home pose's wrist yaw (pointing down)
env.move_tool([box[0], box[1], top + 0.12], tool_quat, timeout=8.0)
env.move_tool([box[0], box[1], top + args.gap], tool_quat, timeout=5.0, tol=0.002)
snap("1_pregrasp")

# Close-ups of the camera hardware itself, shot by moving the (video-only) overview camera around.
import numpy as np  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from scene import look_at_quat, set_pose  # noqa: E402

lay = env.layout
hand = env._hand.get_world_poses()[0].numpy()[0]
cam_z, post_y, top = lay.belt_top_z + 1.15, lay.frame_half_width + 0.15, lay.belt_top_z
overview = UsdGeom.Xformable(env.stage.GetPrimAtPath(CAMERA_PATHS["overview"]))
wrist = np.array(UsdGeom.Xformable(env.stage.GetPrimAtPath(CAMERA_PATHS["wrist"]))
                 .ComputeLocalToWorldTransform(0).ExtractTranslation())
side = (wrist - hand) * (1, 1, 0)  # look at the wrist camera from its own side of the tool
side /= np.linalg.norm(side)
oblique = np.array([side[0] - side[1], side[0] + side[1], 0.0]) / np.sqrt(2)  # 45 deg off, clear of the gantry post
for name, eye, target in (
    ("overhead", (lay.inspect_x - 0.45, -0.35, cam_z - 0.1), (lay.inspect_x, 0.0, cam_z + 0.02)),
    ("side", (lay.inspect_x - 0.4, post_y - 0.45, top + 0.55), (lay.inspect_x, post_y - 0.08, top + 0.39)),
    ("wrist", tuple(wrist + 0.3 * oblique + (0.0, 0.0, 0.12)), tuple(wrist)),
):
    set_pose(overview, eye, look_at_quat(eye, target))
    env.step(30, auto_spawn=False)  # auto-exposure settles
    Image.fromarray(env.capture("overview")).save(out / f"2_mount_{name}_cam.png")
print(f"[snap] box {spec.box_id} size={spec.size} defects={[(m.kind, m.face) for m in spec.defects]}", flush=True)
app.close()

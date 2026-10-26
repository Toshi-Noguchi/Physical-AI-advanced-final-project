"""Runtime side of the inspection cell: conveyor control, box spawning / despawning, the inspection
stop, robot / gripper handles and camera readout.

Import only after ``SimulationApp`` has been created.
"""

from __future__ import annotations

from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.backend as backend_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from box_textures import BoxSpec, sample_box_spec
from boxes import spawn_box
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.surface_gripper import GripperView
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdShade
from scene import (ARM_HOME, CAMERAS, GRIPPER_PATH, HAND_PATH, RAKING_INTENSITY, RAKING_LIGHTS, ROBOT_PATH, TOOL_TIP_Z,
                   Layout, build_scene)

CAMERA_PATHS = {
    "overhead": "/World/Cameras/overhead",
    "side": "/World/Cameras/side",
    "overview": "/World/Cameras/overview",
    "wrist": f"{HAND_PATH}/wrist_cam",
}
BASE_DOFS = ("dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint")
ARM_DOFS = tuple(f"panda_joint{i}" for i in range(1, 8))
TOOL_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # hand +z pointing at the floor (180 deg about world x)
# Finished boxes are parked on the floor far off-stage (see remove_box).
PARK_ORIGIN = np.array([-15.0, 12.0, 0.3])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.r_[0.0, v]), quat_conj(q))[1:]


class InspectionLineEnv:
    def __init__(
        self,
        app,
        out_dir: str | Path,
        seed: int = 0,
        layout: Layout | None = None,
        cameras: tuple[str, ...] = tuple(CAMERA_PATHS),
        p_defect: float = 0.5,
        spawn_interval: float = 5.0,
        resolutions: dict[str, tuple[int, int]] | None = None,
    ):
        self.app = app
        self.rng = np.random.default_rng(seed)
        self.out_dir = Path(out_dir)
        self.layout = layout or Layout()
        self.p_defect = p_defect
        self.spawn_interval = spawn_interval

        SimulationManager.set_physics_sim_device("cpu")
        app.update()
        self.stage = stage_utils.create_new_stage()
        build_scene(self.stage, get_assets_root_path(), self.layout, self.out_dir / "generated")
        while stage_utils.is_stage_loading():
            app.update()
        self._belt = PhysxSchema.PhysxSurfaceVelocityAPI(self.stage.GetPrimAtPath("/World/Line/belt"))
        self._cardboard = UsdShade.Material(self.stage.GetPrimAtPath("/World/PhysicsMaterials/cardboard"))

        resolutions = {**{name: CAMERAS[name][0] for name in cameras}, **(resolutions or {})}
        self.cameras = {
            name: CameraSensor(RtxCamera(CAMERA_PATHS[name]), resolution=resolutions[name], annotators=["rgb"])
            for name in cameras
        }
        self._lights = {  # ambient lights that randomize_lighting varies; the raking lights are switched separately
            prim.GetPath().pathString: (UsdLux.LightAPI(prim), UsdLux.LightAPI(prim).GetIntensityAttr().Get())
            for prim in self.stage.Traverse() if prim.HasAPI(UsdLux.LightAPI)
            and prim.GetPath().pathString not in RAKING_LIGHTS
        }
        self._raking = [UsdLux.LightAPI(self.stage.GetPrimAtPath(p)) for p in RAKING_LIGHTS]
        self.gripper = GripperView(
            paths=f"{GRIPPER_PATH}/gripper",
            max_grip_distance=np.array([0.02]),
            # Generous limits: the stiff arm drives can press the cup into a box resting on the belt, and a
            # grip that breaks at contact never gets to lift (boxes weigh ~1 kg).
            coaxial_force_limit=np.array([5000.0]),
            shear_force_limit=np.array([5000.0]),
            retry_interval=np.array([0.3]),
        )
        app_utils.play()
        app.update()

        self.robot = Articulation(ROBOT_PATH)
        self.dof_index = {name: i for i, name in enumerate(self.robot.dof_names)}
        self._init_robot()

        self.time = SimulationManager.get_simulation_time()
        self.dt = SimulationManager.get_physics_dt()
        self.belt_running = False
        self.suction_on = False
        self.boxes: dict[str, dict] = {}  # prim path -> {"spec", "view", "inspected"}
        self.waiting_box: str | None = None  # box stopped at the inspection station
        self.dropped: list[str] = []
        self._n_spawned = 0
        self._n_parked = 0
        self._last_spawn = -1e9

    def go_home(self, timeout: float = 3.0, tol: float = 0.01) -> bool:
        """Joint-space move back to ARM_HOME (cup pointing down just above the near edge of the line)."""
        self.set_robot_targets(**dict(zip(ARM_DOFS, ARM_HOME)))
        t_end = self.time + timeout
        while self.time < t_end:
            if np.abs(self.arm_q() - ARM_HOME).max() < tol:
                return True
            self.step()
        return False

    # ---------------------------------------------------------------- robot
    def _init_robot(self) -> None:
        q = np.zeros(self.robot.num_dofs)
        for name, value in zip(BASE_DOFS, self.layout.robot_home):
            q[self.dof_index[name]] = value
        for name, value in zip(ARM_DOFS, ARM_HOME):
            q[self.dof_index[name]] = value
        self.robot.set_dof_positions(q)
        self.robot.set_dof_position_targets(q)
        kp, kd = (g.numpy().copy() for g in self.robot.get_dof_gains())
        for name in BASE_DOFS:  # the dummy base joints ship as velocity drives; hold them in place instead
            i = self.dof_index[name]
            kp[:, i], kd[:, i] = 1e7, 1e5
        self.robot.set_dof_gains(stiffnesses=kp, dampings=kd)
        tip = self.stage.GetPrimAtPath(f"{HAND_PATH}/suction_mount/gripper/gripper_tip")
        rng = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeRelativeBound(
            tip, self.stage.GetPrimAtPath(HAND_PATH)).ComputeAlignedRange()
        print(f"[env] suction cup in hand frame: min={rng.GetMin()} max={rng.GetMax()} (TOOL_TIP_Z={TOOL_TIP_Z})")
        self._hand = RigidPrim(HAND_PATH)
        self._link0 = RigidPrim(f"{ROBOT_PATH}/panda_link0")
        self._hand_link = int(self.robot.get_link_indices("panda_hand").numpy()[0])
        self._arm_cols = [self.dof_index[name] for name in ARM_DOFS]

    def set_robot_targets(self, **targets: float) -> None:
        """Set position targets by DOF name, e.g. ``set_robot_targets(panda_joint1=0.3)``."""
        q = self.robot.get_dof_position_targets().numpy().copy()
        for name, value in targets.items():
            q[:, self.dof_index[name]] = value
        self.robot.set_dof_position_targets(q)

    def set_base_pose(self, x: float, y: float, yaw: float) -> None:
        """Teleport the Ridgeback (its x / y / yaw dummy joints) and hold it there."""
        self._teleport(dict(zip(BASE_DOFS, (x, y, yaw))))

    def set_arm(self, q) -> None:
        """Teleport the arm joints and hold them there."""
        self._teleport(dict(zip(ARM_DOFS, map(float, q))))

    def _teleport(self, values: dict[str, float]) -> None:
        q = self.robot.get_dof_positions().numpy().copy()
        targets = self.robot.get_dof_position_targets().numpy().copy()
        for name, value in values.items():
            q[:, self.dof_index[name]] = targets[:, self.dof_index[name]] = value
        self.robot.set_dof_positions(q)
        self.robot.set_dof_velocities(np.zeros_like(q))
        self.robot.set_dof_position_targets(targets)

    def arm_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and 3x3 rotation of panda_link0 (the frame of panda_kin)."""
        pos, quat = self._link0.get_world_poses()
        w, x, y, z = quat.numpy()[0]
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return pos.numpy()[0], rot

    def randomize_lighting(self, rng: np.random.Generator, spread: float = 0.3) -> None:
        """Scale every light's intensity by up to +-spread and pick a common colour temperature."""
        kelvin = float(rng.uniform(4000.0, 6500.0))
        for light, base in self._lights.values():
            light.GetIntensityAttr().Set(float(base * rng.uniform(1 - spread, 1 + spread)))
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(kelvin)

    def set_inspection_light(self, on: bool, ambient: float = 0.4) -> None:
        """Raking lights on, and the hall dimmed to ``ambient`` of its level (as under an inspection hood)."""
        for light in self._raking:
            light.GetIntensityAttr().Set(RAKING_INTENSITY if on else 0.0)
        for light, _ in self._lights.values():
            attr = light.GetIntensityAttr()
            attr.Set(attr.Get() * (ambient if on else 1.0 / ambient))

    def capture_inspection(self) -> tuple[np.ndarray, np.ndarray]:
        """Top and side inspection images, taken with the raking lights flashed on."""
        self.set_inspection_light(True)
        self.step(3, auto_spawn=False)  # let the renderer settle on the new lighting
        top, side = self.capture("overhead"), self.capture("side")
        self.set_inspection_light(False)
        self.step(1, auto_spawn=False)
        return top, side

    def arm_q(self) -> np.ndarray:
        return self.robot.get_dof_positions().numpy()[0, self._arm_cols]

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the suction cup face: position and (w, x, y, z) quaternion."""
        pos, quat = self._hand.get_world_poses()
        pos, quat = pos.numpy()[0], quat.numpy()[0]
        return pos + quat_rotate(quat, np.array([0.0, 0.0, TOOL_TIP_Z])), quat

    def ik_step(self, target_pos, target_quat, max_step: float = 0.05, damping: float = 0.05) -> float:
        """One damped-least-squares step of the arm toward a cup-face pose; returns the position error."""
        tip, quat = self.ee_pose()
        hand_pos = self._hand.get_world_poses()[0].numpy()[0]
        jac = self.robot.get_jacobian_matrices().numpy()[0, self._hand_link - 1][:, self._arm_cols]
        r = tip - hand_pos  # the Jacobian is for the hand origin: shift its linear rows to the cup face
        skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        jac = np.vstack([jac[:3] - skew @ jac[3:], jac[3:]])
        q_err = quat_mul(np.asarray(target_quat, float), quat_conj(quat))
        err = np.concatenate([np.asarray(target_pos, float) - tip, 2.0 * q_err[1:] * np.sign(q_err[0] or 1.0)])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
        dq *= min(1.0, max_step / (np.abs(dq).max() + 1e-9))
        q = self.robot.get_dof_positions().numpy()[0]
        self.set_robot_targets(**{name: q[col] + d for name, col, d in zip(ARM_DOFS, self._arm_cols, dq)})
        return float(np.linalg.norm(err[:3]))

    def move_tool(self, target_pos, target_quat=None, timeout: float = 4.0, tol: float = 0.003) -> bool:
        """Servo the cup face to ``target_pos`` (default orientation: pointing straight down)."""
        target_quat = TOOL_DOWN if target_quat is None else target_quat
        t_end = self.time + timeout
        while self.time < t_end:
            if self.ik_step(target_pos, target_quat) < tol:
                return True
            self.step()
        return False

    # ---------------------------------------------------------------- line
    def set_belt(self, running: bool) -> None:
        self.belt_running = running
        speed = self.layout.belt_speed if running else 0.0
        self._belt.GetSurfaceVelocityAttr().Set(Gf.Vec3f(speed, 0.0, 0.0))

    def spawn_box(self, spec: BoxSpec | None = None, x: float | None = None, max_yaw_deg: float = 4.0) -> str:
        """Drop a new box onto the belt at ``x`` (default: the upstream spawn point)."""
        self._n_spawned += 1
        box_id = f"box_{self._n_spawned:04d}"
        spec = spec or sample_box_spec(self.rng, box_id, p_defect=self.p_defect)
        spec.box_id = box_id
        lay = self.layout
        x = lay.spawn_x if x is None else x
        pos = (x, float(self.rng.uniform(-0.03, 0.03)), lay.belt_top_z + spec.size[2] / 2 + 0.003)
        path = spawn_box(self.stage, spec, f"/World/Boxes/{box_id}", self.out_dir / "textures", pos,
                         yaw_deg=float(self.rng.uniform(-max_yaw_deg, max_yaw_deg)),
                         mass=float(self.rng.uniform(0.4, 1.2)),
                         physics_mat=self._cardboard)
        self.boxes[path] = {"spec": spec, "view": None, "inspected": False, "on_line": True}
        self._last_spawn = self.time
        return path

    def box_pose(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        entry = self.boxes[path]
        if entry["view"] is None:
            entry["view"] = RigidPrim(path)
        pos, quat = entry["view"].get_world_poses()
        return pos.numpy()[0], quat.numpy()[0]

    def remove_box(self, path: str) -> None:
        """Take a box off the line. Deleting a body mid-run invalidates every PhysX tensor view (the robot's
        included), so the box is hidden and parked on the floor off-stage instead."""
        entry = self.boxes.pop(path, None)
        if self.waiting_box == path:
            self.waiting_box = None
        if entry is None:
            return
        view = RigidPrim(path)
        k = self._n_parked
        self._n_parked += 1
        pose = {"positions": (PARK_ORIGIN + (0.6 * (k % 20), 0.6 * (k // 20), 0.0))[None],
                "orientations": [[1.0, 0.0, 0.0, 0.0]]}
        # Move it in PhysX (the default USD backend alone is ignored by a running simulation) *and* in USD: when
        # new boxes are spawned in the same frame, PhysX re-reads poses from USD and would put it back.
        with backend_utils.use_backend("tensor", raise_on_fallback=True):
            view.set_world_poses(**pose)
            view.set_velocities(linear_velocities=np.zeros((1, 3)), angular_velocities=np.zeros((1, 3)))
        with backend_utils.use_backend("usd"):
            view.set_world_poses(**pose)
        UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).MakeInvisible()

    def take_off_line(self, path: str) -> None:
        """The box left the conveyor (picked as NG): stop treating it as line traffic and free the station."""
        self.boxes[path]["on_line"] = False
        if self.waiting_box == path:
            self.waiting_box = None

    def release_box(self) -> None:
        """Let the box at the station continue downstream (inspection passed / handled)."""
        self.waiting_box = None
        self.set_belt(True)

    def _update_line(self, auto_spawn: bool) -> None:
        lay = self.layout
        for path in list(self.boxes):
            entry = self.boxes[path]
            if not entry["on_line"]:  # picked off the line (held, or sitting on the NG table)
                continue
            if entry["view"] is None:  # physics picks new bodies up on the next step
                entry["view"] = RigidPrim(path)
                continue
            pos, _ = self.box_pose(path)
            if pos[0] > lay.despawn_x:
                self.remove_box(path)
            elif pos[2] < lay.belt_top_z - 0.25 and path != self.held_box():
                self.dropped.append(path)  # fell off the line
                self.remove_box(path)
            elif (self.belt_running and self.waiting_box is None and not entry["inspected"]
                  and pos[0] >= lay.inspect_x):
                entry["inspected"] = True
                self.waiting_box = path
                self.set_belt(False)
        if auto_spawn and self.belt_running and self.time - self._last_spawn >= self.spawn_interval:
            clear = all(self.box_pose(p)[0][0] > lay.spawn_x + 0.6 for p in self.boxes
                        if self.boxes[p]["view"] and self.boxes[p]["on_line"])
            if clear:
                self.spawn_box()

    # ---------------------------------------------------------------- gripper
    def set_suction(self, on: bool) -> None:
        self.suction_on = on
        self.gripper.apply_gripper_action([1.0 if on else -1.0])

    def release(self, timeout_steps: int = 120) -> None:
        """Switch the suction off and wait until the box is really free (the attachment lingers a few frames
        after the gripper reports it open; a box teleported before that gets yanked across the scene)."""
        self.set_suction(False)
        for _ in range(timeout_steps):
            if not self.gripper.get_gripped_objects()[0]:
                break
            self.step(1, auto_spawn=False)
        else:
            raise RuntimeError("the surface gripper did not release")
        self.step(3, auto_spawn=False)

    def held_box(self) -> str | None:
        objs = self.gripper.get_gripped_objects()[0]
        for path in self.boxes:
            if any(str(o).startswith(path) for o in objs):
                return path
        return None

    # ---------------------------------------------------------------- DROID-style interface (MolmoAct2-DROID)
    def droid_state(self) -> np.ndarray:
        """DROID ``observation.state``: 7 arm joint positions + gripper position (0 open .. 1 closed).

        The cup reports like a parallel gripper: closed on an object (0.5) or on nothing (0.8)."""
        grip = 0.0 if not self.suction_on else (0.5 if self.held_box() else 0.8)
        return np.r_[self.arm_q(), grip].astype(np.float32)

    def apply_droid_action(self, action) -> None:
        """Absolute joint targets; the suction switches on when the gripper command says "close"."""
        self.set_robot_targets(**dict(zip(ARM_DOFS, map(float, action[:7]))))
        if (action[7] > 0.5) != self.suction_on:
            self.set_suction(bool(action[7] > 0.5))

    # ---------------------------------------------------------------- stepping / sensing
    def step(self, n: int = 1, auto_spawn: bool = True) -> None:
        for _ in range(n):
            self._update_line(auto_spawn)
            self.app.update()
            self.time = SimulationManager.get_simulation_time()

    def capture(self, name: str) -> np.ndarray | None:
        data, _ = self.cameras[name].get_data("rgb")
        if data is None:
            return None
        img = data.numpy()
        return img[..., :3].astype(np.uint8) if img.ndim == 3 and img.size else None

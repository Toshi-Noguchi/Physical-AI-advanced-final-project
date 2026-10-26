"""Rule-based skills around the grasp: carry the held box to the reject (NG) table, put it down, and go back to
the line. Paths are smooth (Catmull-Rom curves with min-jerk timing) and arm motions are planned kinematically
with panda_kin, like the grasp demonstrations.

The Ridgeback is omnidirectional, so it keeps facing +y (as at the line) and just translates: back off the line,
swing wide of the table, and come up to one of the drop slots along the table's -y edge.
"""

from __future__ import annotations

import numpy as np
import panda_kin as pk
from PIL import Image
from env import ARM_DOFS, BASE_DOFS, InspectionLineEnv
from expert import min_jerk, plan_pick, slerp
from grasp_target import box_in_link0, keep_out_spots
from scene import ARM_HOME, BASE_HALF_LENGTH, TOOL_TIP_Z, Layout

CARRY_TIP = np.array([0.30, 0.0, 0.55])  # cup in the panda_link0 frame while driving: high and tucked in
INSPECT_TIP = np.array([0.20, 0.0, 0.55])  # cup while the station cameras look: pulled back, clear of the box
DROP_REACH = 0.5  # panda_link0 -> drop point along the robot's heading, as at the line
TABLE_GAP = 0.07  # base front bumper -> table edge when placing
BASE_HALF_WIDTH = 0.40  # the Ridgeback is 0.79 m wide
SWING = 0.45  # how far below (-y) the table edge the base travels before coming up to a slot
DRIVE_SPEED = 0.35  # m/s, average along the path (min-jerk peaks at ~1.9x)
BASE_MAX_EFFORT = (2e4, 2e4, 1e4)  # N, N, N m for the x / y / yaw dummy joints
ARM_SPEED = 0.2  # m/s, average for cup moves


def catmull_rom(points: np.ndarray, samples: int = 400) -> np.ndarray:
    """Dense polyline of a centripetal-free (uniform) Catmull-Rom curve through ``points`` (k x d)."""
    p = np.vstack([points[0], points, points[-1]])
    out = []
    for i in range(len(points) - 1):
        p0, p1, p2, p3 = p[i:i + 4]
        t = np.linspace(0, 1, max(2, samples // (len(points) - 1)), endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    out.append(points[-1:])
    return np.vstack(out)


def along(poly: np.ndarray, duration: float, dt: float) -> np.ndarray:
    """Sample a polyline at the physics rate with min-jerk timing over ``duration``."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    n = max(1, int(round(duration / dt)))
    s = min_jerk(np.arange(1, n + 1) / n) * cum[-1]
    return np.stack([np.interp(s, cum, poly[:, j]) for j in range(poly.shape[1])], axis=1)


def follow(env: InspectionLineEnv, arm=None, base=None, on_step=None, settle: int = 5) -> None:
    """Track per-physics-step arm joint and/or base (x, y, yaw) targets."""
    n = len(arm) if arm is not None else len(base)
    for k in range(n + settle):
        targets = {}
        if arm is not None:
            targets.update(zip(ARM_DOFS, map(float, arm[min(k, n - 1)])))
        if base is not None:
            targets.update(zip(BASE_DOFS, map(float, base[min(k, n - 1)])))
        env.set_robot_targets(**targets)
        env.step(1, auto_spawn=False)
        if on_step:
            on_step(env)


def move_cup(env: InspectionLineEnv, waypoints, rot=None, speed: float = ARM_SPEED, on_step=None) -> None:
    """Move the cup through ``waypoints`` (panda_link0 frame) on a smooth curve, turning to ``rot`` (3x3)."""
    q = env.arm_q()
    start = pk.fk(q, TOOL_TIP_Z)[0]
    poly = catmull_rom(np.vstack([start[:3, 3], *waypoints]))
    length = np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()
    pos = along(poly, max(0.8, length / speed), env.dt)
    rot = start[:3, :3] if rot is None else rot
    turn = min_jerk(np.arange(1, len(pos) + 1) / len(pos))
    traj = np.empty((len(pos), 7))
    for k, p in enumerate(pos):
        r = slerp(start[:3, :3], rot, turn[k])
        q = pk.ik(pk.null_step(q, TOOL_TIP_Z, ARM_HOME, 1.0 * env.dt), p, r, TOOL_TIP_Z, iters=15)
        err_p, err_r = pk.pose_error(q, p, r, TOOL_TIP_Z)
        if err_p > 0.005 or err_r > np.radians(3):
            raise RuntimeError(f"cup path out of reach near {np.round(p, 3)} ({err_p * 1000:.0f} mm)")
        traj[k] = q
    follow(env, arm=traj, on_step=on_step)


def arm_home(env: InspectionLineEnv, duration: float = 1.5, on_step=None, tol: float = 0.005) -> None:
    """Joint-space move to ARM_HOME, the pose the grasp policy's demonstrations start from; waits until there."""
    q0 = env.arm_q()
    s = min_jerk(np.arange(1, int(round(duration / env.dt)) + 1) / round(duration / env.dt))[:, None]
    follow(env, arm=q0 + (np.asarray(ARM_HOME) - q0) * s, on_step=on_step)
    for _ in range(120):
        if np.abs(env.arm_q() - ARM_HOME).max() < tol:
            return
        env.step(1, auto_spawn=False)
        if on_step:
            on_step(env)
    raise RuntimeError(f"arm did not settle at home: {env.arm_q().round(3)}")


def retract_for_inspection(env: InspectionLineEnv, on_step=None) -> None:
    """Pull the cup back over the base so the arm does not hide the box from the station cameras."""
    rot_home = pk.fk(np.asarray(ARM_HOME), TOOL_TIP_Z)[0][:3, :3]
    move_cup(env, [INSPECT_TIP], rot_home, on_step=on_step)


def drive(env: InspectionLineEnv, waypoints, yaw: float | None = None, speed: float = DRIVE_SPEED, on_step=None):
    """Drive the base through world (x, y) waypoints, ending at ``yaw`` (default: keep the heading)."""
    cols = [env.dof_index[n] for n in BASE_DOFS]
    # The asset caps the dummy base drives at 1 kN, less than the chassis' friction on the floor: the base
    # would crawl behind its targets. Give them enough authority to follow the path.
    env.robot.set_dof_max_efforts(np.array(BASE_MAX_EFFORT), dof_indices=cols)
    start = env.robot.get_dof_positions().numpy()[0, cols]
    poly = catmull_rom(np.vstack([start[:2], *waypoints]))
    length = np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()
    xy = along(poly, max(1.0, length / speed), env.dt)
    yaw = start[2] if yaw is None else yaw
    yaws = start[2] + (yaw - start[2]) * min_jerk(np.arange(1, len(xy) + 1) / len(xy))
    follow(env, base=np.c_[xy, yaws], on_step=on_step, settle=0)
    goal = np.r_[xy[-1], yaw]
    for _ in range(120):  # arrive before the arm starts working
        if np.abs(env.robot.get_dof_positions().numpy()[0, cols] - goal).max() < 0.003:
            return
        env.step(1, auto_spawn=False)
        if on_step:
            on_step(env)
    raise RuntimeError(f"base stopped short of {goal.round(3)}: {env.robot.get_dof_positions().numpy()[0, cols].round(3)}")


class RejectTable:
    """Drop slots along the robot-side (-y) edge of the NG table. Empty slots fill first, then boxes stack."""

    def __init__(self, layout: Layout, n_slots: int = 4):
        cx, cy = layout.ng_table_center
        sx, sy, h = layout.ng_table_size
        self.layout = layout
        self.x_min, self.edge_y = cx - sx / 2, cy - sy / 2
        self.slot_x = [cx - sx / 2 + sx / n_slots * (k + 0.5) for k in range(n_slots)]
        self.stack_top = [h] * n_slots
        self.count = [0] * n_slots

    def next_slot(self) -> int:
        return int(np.argmin(self.count))

    def base_pose(self, slot: int) -> tuple[float, float, float]:
        return self.slot_x[slot], self.edge_y - TABLE_GAP - BASE_HALF_LENGTH, np.pi / 2

    def route_to(self, slot: int) -> list[tuple[float, float]]:
        """Back off the line, pass west of the table, then come up to the slot from below (-y)."""
        hx, hy, _ = self.layout.robot_home
        bx, by, _ = self.base_pose(slot)
        clear_x = self.x_min - BASE_HALF_WIDTH - 0.25
        return [(hx, hy - 0.7), (clear_x, by - SWING), (bx, by - SWING), (bx, by)]

    def route_back(self, slot: int) -> list[tuple[float, float]]:
        hx, hy, _ = self.layout.robot_home
        return [*self.route_to(slot)[-2::-1], (hx, hy)]


def carry_to_table(env: InspectionLineEnv, table: RejectTable, on_step=None) -> int:
    """Lift the held box into the carry pose and drive up to the next free slot. Returns the slot."""
    rot_home = pk.fk(np.asarray(ARM_HOME), TOOL_TIP_Z)[0][:3, :3]
    move_cup(env, [CARRY_TIP], rot_home, on_step=on_step)
    slot = table.next_slot()
    *_, yaw = table.base_pose(slot)
    drive(env, table.route_to(slot), yaw, on_step=on_step)
    return slot


def place(env: InspectionLineEnv, table: RejectTable, slot: int, path: str, on_step=None) -> None:
    """Lower the held box onto the slot (or the stack in it), let go, and pull the arm back up."""
    height = env.boxes[path]["spec"].size[2]
    base_pos, base_rot = env.arm_base_pose()
    drop = np.array([table.slot_x[slot], base_pos[1] + DROP_REACH, table.stack_top[slot] + height + 0.004])
    drop = base_rot.T @ (drop - base_pos)  # world -> panda_link0 frame
    move_cup(env, [drop + (0, 0, 0.12), drop], on_step=on_step)
    env.release()
    table.stack_top[slot] += height
    table.count[slot] += 1
    move_cup(env, [drop + (0, 0, 0.12), CARRY_TIP], on_step=on_step)


def return_to_line(env: InspectionLineEnv, table: RejectTable, slot: int, on_step=None) -> None:
    drive(env, table.route_back(slot), env.layout.robot_home[2], on_step=on_step)
    arm_home(env, on_step=on_step)


# The grasp policy's cameras, at the resolutions its demonstrations were recorded with (same fields of view as the
# inspection shots, so a full-resolution capture is simply scaled down).
VLA_VIEWS = {"overhead": (480, 640), "side": (480, 640), "wrist": (360, 640)}


def vla_views(env: InspectionLineEnv) -> list[np.ndarray]:
    images = []
    for name, (h, w) in VLA_VIEWS.items():
        img = env.capture(name)
        if img.shape[:2] != (h, w):
            img = np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.BOX))
        images.append(img)
    return images


def vla_pick(env: InspectionLineEnv, client, task: str, max_seconds: float = 15.0, exec_steps: int = 25,
             fps: int = 30, on_step=None) -> dict:
    """Run the grasp policy from the home pose: it drives the arm (absolute joint targets) and the suction itself.
    Like its demonstrations, an episode ends when the arm has left home and come back with the suction on (held for
    0.5 s), or on timeout. Whether the box was really picked is for the caller to check."""
    substeps = max(1, int(round(1.0 / (fps * env.dt))))
    t0, queries, left_home, back_since = env.time, 0, False, None
    while env.time - t0 < max_seconds:
        actions = client.act(vla_views(env), task, env.droid_state())["actions"]
        queries += 1
        for a in actions[:exec_steps]:
            env.apply_droid_action(a)
            env.step(substeps, auto_spawn=False)
            if on_step:
                on_step(env)
            near_home = np.abs(env.arm_q() - ARM_HOME).max() < 0.15
            left_home |= not near_home
            if left_home and near_home and env.suction_on:
                back_since = env.time if back_since is None else back_since
                if env.time - back_since >= 0.5:
                    return {"finished": True, "queries": queries, "seconds": round(env.time - t0, 2)}
            else:
                back_since = None
    return {"finished": False, "queries": queries, "seconds": round(env.time - t0, 2)}


def scripted_pick(env: InspectionLineEnv, path: str, rng: np.random.Generator, on_step=None) -> bool:
    """Privileged fallback for the grasp (the expert that generated the VLA's demonstrations): pick the box at
    the station and carry it back to the home pose. Returns whether the box is held."""
    center, box_rot, spec = box_in_link0(env, path)
    for _ in range(30):
        plan, _ = plan_pick(rng, env.arm_q(), np.asarray(ARM_HOME), center, box_rot, spec.size, keep_out_spots(spec),
                            TOOL_TIP_Z, env.dt)
        if plan is not None:
            break
    else:
        return False
    for k in range(len(plan.q)):
        env.set_robot_targets(**dict(zip(ARM_DOFS, map(float, plan.q[k]))))
        if (plan.grip[k] > 0.5) != env.suction_on:
            env.set_suction(bool(plan.grip[k] > 0.5))
        env.step(1, auto_spawn=False)
        if on_step:
            on_step(env)
    return env.held_box() == path

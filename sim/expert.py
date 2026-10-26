"""Scripted suction-grasp expert that generates the demonstrations for fine-tuning MolmoAct2.

It is privileged (reads the box pose from the simulator) and plans kinematically with :mod:`panda_kin` at the
physics rate, so the recorded joint targets are smooth and do not inherit the simulator's tracking lag.

Every episode draws its own variation so the data is not one IK solution replayed 500 times: grasp point and
wrist yaw, approach curve and hover height, durations, a small tilt on the way, lift direction, the elbow
posture (nullspace), and optionally smooth joint noise in execution. That noise is DART-style: the executed
targets are perturbed while the recorded actions stay on the plan, so the data also shows how to get back on
track.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import panda_kin as pk

PHASES = ("approach", "grip", "lift", "return", "hold")
MAX_JOINT_SPEED = 1.0  # rad/s; DROID-like teleop speeds, well under the Panda's limits


def min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)


def bezier(ctrl: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Cubic Bezier through 4 control points, sampled at ``s`` in [0, 1]."""
    s = s[:, None]
    return (1 - s) ** 3 * ctrl[0] + 3 * (1 - s) ** 2 * s * ctrl[1] + 3 * (1 - s) * s**2 * ctrl[2] + s**3 * ctrl[3]


def axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    rv = pk.rot_error(r1, r0)
    angle = np.linalg.norm(rv)
    return r0 if angle < 1e-9 else axis_angle(rv, s * angle) @ r0


def ou_noise(rng: np.random.Generator, n: int, dt: float, sigma: float, tau: float = 0.5) -> np.ndarray:
    """Smooth (Ornstein-Uhlenbeck) joint noise, stationary std ``sigma``."""
    x = np.zeros((n, 7))
    a, b = np.exp(-dt / tau), sigma * np.sqrt(1 - np.exp(-2 * dt / tau))
    for k in range(1, n):
        x[k] = a * x[k - 1] + b * rng.standard_normal(7)
    return x


@dataclass
class GraspPlan:
    dt: float
    q: np.ndarray  # (N, 7) joint targets: q[k] is where the arm should be at time (k + 1) * dt
    grip: np.ndarray  # (N,) suction command, 0 / 1
    phase: np.ndarray  # (N,) index into PHASES
    noise: np.ndarray  # (N, 7) added to the executed targets only (zeros when DART is off)
    info: dict = field(default_factory=dict)


def plan_pick(
    rng: np.random.Generator,
    q_start: np.ndarray,
    q_home: np.ndarray,
    box_center: np.ndarray,
    box_rot: np.ndarray,
    box_size: tuple[float, float, float],
    keep_out: list[tuple[float, float, float]],
    tip: float,
    dt: float,
    p_dart: float = 0.5,
) -> tuple[GraspPlan | None, str]:
    """Plan approach -> grip -> lift -> return home -> hold for the box. All poses are in the panda_link0 frame.

    ``keep_out`` lists (x, y, radius) spots on the top face, in the box frame, that the cup must avoid (dents,
    tears). Returns (None, reason) when the sampled plan is infeasible (the caller just samples again)."""
    length, width, height = box_size
    info: dict = {}

    # Grasp point on the top face, clear of the edges and of damage.
    for _ in range(50):
        u = rng.uniform(-1, 1) * max(0.0, length / 2 - 0.06) * 0.7
        v = rng.uniform(-1, 1) * max(0.0, width / 2 - 0.06) * 0.7
        if all(np.hypot(u - x, v - y) > r + 0.04 for x, y, r in keep_out):
            break
    else:
        return None, "no clear grasp point"
    grasp = box_center + box_rot @ np.array([u, v, height / 2])
    # Stop a few mm short of the top: the surface gripper looks for the box with a short ray from the cup
    # face, which misses if the face is pushed into the box.
    contact = grasp + np.array([0.0, 0.0, rng.uniform(0.002, 0.005)])

    # Tool orientation: pointing down like the home pose, with a random yaw about the vertical.
    rot_home = pk.fk(q_home, tip)[0][:3, :3]
    yaw = rng.uniform(-np.radians(35), np.radians(35))
    rot_grasp = axis_angle((0, 0, 1), yaw) @ rot_home
    start_pose = pk.fk(q_start, tip)[0]
    p_start, rot_start = start_pose[:3, 3], start_pose[:3, :3]

    # Approach: one curve from the start to the contact point, ending with a vertical descent.
    hover = rng.uniform(0.07, 0.16)
    above = contact + (0.0, 0.0, hover)
    c1 = p_start + 0.4 * (above - p_start) + (rng.normal(0, 0.04), rng.normal(0, 0.04), rng.uniform(0.0, 0.08))
    t_approach = rng.uniform(2.0, 3.2)
    tilt_axis = axis_angle((0, 0, 1), rng.uniform(0, 2 * np.pi)) @ np.array([1.0, 0.0, 0.0])
    tilt = rng.uniform(0.0, np.radians(8))
    n_a = int(round(t_approach / dt))
    tau = np.arange(1, n_a + 1) / n_a
    pos = [bezier(np.array([p_start, c1, above, contact]), min_jerk(tau))]
    rots = []
    for t in tau:
        s_rot = float(min_jerk(np.array([t / 0.75]))[0])  # finish turning before the descent
        rots.append(axis_angle(tilt_axis, tilt * np.sin(np.pi * s_rot)) @ slerp(rot_start, rot_grasp, s_rot))

    # Grip: hold still on the box while the vacuum builds up.
    t_grip = rng.uniform(0.25, 0.5)
    n_g = int(round(t_grip / dt))
    pos.append(np.repeat(contact[None], n_g, 0))
    rots += [rot_grasp] * n_g

    # Lift: vertical take-off curving toward a slightly offset point above, with a little wrist yaw.
    rise = rng.uniform(0.12, 0.2)
    lift_end = contact + (rng.normal(0, 0.02), rng.normal(0, 0.02), rise)
    t_lift = rng.uniform(1.1, 1.6)
    n_l = int(round(t_lift / dt))
    tau_l = np.arange(1, n_l + 1) / n_l
    ctrl = np.array([contact, contact + (0, 0, 0.4 * rise), lift_end - (0, 0, 0.2 * rise), lift_end])
    pos.append(bezier(ctrl, min_jerk(tau_l)))
    lift_yaw = rng.uniform(-np.radians(10), np.radians(10))
    rots += [axis_angle((0, 0, 1), lift_yaw * s) @ rot_grasp for s in min_jerk(tau_l)]

    # Return: carry the box back to the home pose on an arc that keeps it clear of the line.
    home_pose = pk.fk(q_home, tip)[0]
    p_home, rot_home = home_pose[:3, 3], home_pose[:3, :3]
    arc = rng.uniform(0.02, 0.08)
    t_return = rng.uniform(1.8, 2.8)
    n_r = int(round(t_return / dt))
    tau_r = np.arange(1, n_r + 1) / n_r
    ctrl = np.array([lift_end, lift_end + (0, 0, arc), p_home + (0, 0, arc), p_home])
    pos.append(bezier(ctrl, min_jerk(tau_r)))
    rot_lifted = rots[-1]
    rots += [slerp(rot_lifted, rot_home, s) for s in min_jerk(tau_r)]

    t_hold = rng.uniform(0.3, 0.6)
    n_h = int(round(t_hold / dt))
    pos.append(np.repeat(p_home[None], n_h, 0))
    rots += [rot_home] * n_h

    pos = np.concatenate(pos)
    phase = np.repeat(np.arange(len(PHASES)), [n_a, n_g, n_l, n_r, n_h])
    t_on = t_approach - rng.uniform(0.0, 0.25)  # vacuum on just before or at contact
    grip = (np.arange(1, len(pos) + 1) * dt >= t_on).astype(np.float32)

    # Track the path kinematically. The elbow drifts slowly toward a per-episode posture, then back to the
    # home posture while returning, so every episode ends in the same joint configuration.
    q_rest = q_home + rng.normal(0, [0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0])
    t = np.arange(1, len(pos) + 1) * dt
    t_back = t_approach + t_grip + t_lift
    to_home = min_jerk((t - t_back) / t_return)
    q = np.asarray(q_start, float)
    traj = np.empty((len(pos), 7))
    near_box = phase >= 1
    near_box[:n_a] = t[:n_a] > t_approach - 0.5
    for k, (p, r) in enumerate(zip(pos, rots)):
        rest = q_rest + (q_home - q_rest) * to_home[k]
        q = pk.null_step(q, tip, rest, gain=(0.5 + 2.5 * to_home[k]) * dt)
        q = pk.ik(q, p, r, tip, iters=15)
        err_p, err_r = pk.pose_error(q, p, r, tip)
        if err_p > (0.002 if near_box[k] else 0.006) or err_r > np.radians(2):
            return None, f"ik error {err_p * 1000:.1f} mm / {np.degrees(err_r):.1f} deg at step {k} ({PHASES[phase[k]]})"
        traj[k] = q
    speed = np.abs(np.diff(np.vstack([q_start, traj]), axis=0)).max() / dt
    if speed > MAX_JOINT_SPEED:
        return None, f"joint speed {speed:.2f} rad/s"

    # DART: smooth noise while moving freely, faded out well before contact, and small while carrying the box.
    noise = np.zeros_like(traj)
    sigma = rng.uniform(0.005, 0.02) if rng.random() < p_dart else 0.0
    if sigma > 0:
        fade_in = np.clip(t / 0.4, 0, 1)
        fade_out = np.clip((t_approach - 0.8 - t) / 0.4, 0, 1)
        carry = 0.5 * np.sin(np.pi * np.clip((t - t_approach - t_grip) / (t_lift + t_return), 0, 1))
        noise = ou_noise(rng, len(pos), dt, sigma) * (fade_in * fade_out + carry)[:, None]

    info.update(
        grasp_uv=(float(u), float(v)), wrist_yaw=float(yaw), hover=float(hover), tilt=float(tilt),
        t_approach=float(t_approach), t_grip=float(t_grip), t_lift=float(t_lift), t_return=float(t_return),
        t_hold=float(t_hold), rise=float(rise), arc=float(arc), dart_sigma=float(sigma), q_rest=q_rest.tolist(),
        max_joint_speed=float(speed), end_q_error=float(np.abs(traj[-1] - q_home).max()),
    )
    info["contact"] = contact.tolist()
    return GraspPlan(dt=dt, q=traj, grip=grip, phase=phase, noise=noise, info=info), "ok"

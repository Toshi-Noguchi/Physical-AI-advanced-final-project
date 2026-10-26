"""Franka Panda kinematics in numpy (forward kinematics, Jacobian, damped least-squares IK).

Used by the scripted grasp expert to plan joint trajectories purely kinematically, so the recorded actions
are smooth and do not inherit the simulator's tracking lag. Frames follow franka_description: modified DH
(Craig) from panda_link0 to the flange, then panda_hand rotated -45 deg about z. Poses are in the
panda_link0 frame; the tool point is ``tip`` metres along panda_hand +z.
"""

from __future__ import annotations

import numpy as np

# (a, d, alpha) per joint, modified DH; the flange adds d = 0.107.
_DH = np.array([
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
])
FLANGE_D = 0.107
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


_HAND = _dh(0.0, FLANGE_D, 0.0, -np.pi / 4)  # flange + panda_hand's -45 deg


def fk(q: np.ndarray, tip: float = 0.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """4x4 pose of the tool point and the joint frames (for the Jacobian)."""
    t = np.eye(4)
    frames = []
    for (a, d, alpha), theta in zip(_DH, q):
        t = t @ _dh(a, d, alpha, theta)
        frames.append(t)
    t = t @ _HAND
    t[:3, 3] += t[:3, 2] * tip
    return t, frames


def jacobian(q: np.ndarray, tip: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Geometric 6x7 Jacobian (linear; angular) of the tool point, plus its pose."""
    pose, frames = fk(q, tip)
    p = pose[:3, 3]
    jac = np.zeros((6, 7))
    for i, f in enumerate(frames):  # modified DH: joint i rotates about z of frame i
        z, o = f[:3, 2], f[:3, 3]
        jac[:3, i] = np.cross(z, p - o)
        jac[3:, i] = z
    return jac, pose


def rot_error(r_target: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Rotation vector taking ``r`` to ``r_target`` (both 3x3, same frame)."""
    e = r_target @ r.T
    angle = np.arccos(np.clip((np.trace(e) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def ik_step(q, pos, rot, tip: float, damping: float = 0.02) -> np.ndarray:
    """One damped least-squares step toward the tool pose (pos, 3x3 rot); clipped to the joint limits."""
    jac, pose = jacobian(q, tip)
    err = np.concatenate([pos - pose[:3, 3], rot_error(rot, pose[:3, :3])])
    dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
    return np.clip(q + dq, Q_MIN, Q_MAX)


def ik(q0, pos, rot, tip: float, iters: int = 30, tol: float = 1e-6) -> np.ndarray:
    q = np.asarray(q0, float)
    for _ in range(iters):
        q_new = ik_step(q, pos, rot, tip)
        if np.abs(q_new - q).max() < tol:
            return q_new
        q = q_new
    return q


def null_step(q, tip: float, q_rest, gain: float) -> np.ndarray:
    """Move a fraction ``gain`` toward ``q_rest`` within the Jacobian's nullspace (the elbow self-motion), so
    the tool pose stays put."""
    jac, _ = jacobian(q, tip)
    proj = np.eye(7) - np.linalg.pinv(jac) @ jac
    return np.clip(q + proj @ (gain * (np.asarray(q_rest) - q)), Q_MIN, Q_MAX)


def pose_error(q, pos, rot, tip: float) -> tuple[float, float]:
    """Position (m) and orientation (rad) error of the tool at ``q``."""
    pose, _ = fk(q, tip)
    return float(np.linalg.norm(pos - pose[:3, 3])), float(np.linalg.norm(rot_error(rot, pose[:3, :3])))

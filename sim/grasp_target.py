"""Where the box to grasp is, in the arm's frame, and which spots on its top the suction cup must avoid."""

from __future__ import annotations

import numpy as np
from box_textures import BoxSpec, face_point


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def box_in_link0(env, path: str) -> tuple[np.ndarray, np.ndarray, BoxSpec]:
    """Centre and 3x3 orientation of the box in the panda_link0 frame, plus its spec."""
    box_pos, box_quat = env.box_pose(path)
    base_pos, base_rot = env.arm_base_pose()
    return base_rot.T @ (box_pos - base_pos), base_rot.T @ quat_to_rot(box_quat), env.boxes[path]["spec"]


def keep_out_spots(spec: BoxSpec) -> list[tuple[float, float, float]]:
    """Damage on the top face, as (x, y, radius) in the box frame: the cup must not seal over it."""
    spots = []
    for m in spec.defects:
        if m.face == "pz" and m.kind in ("dent", "tear"):
            x, y, _ = face_point(spec.size, "pz", m.uv)
            spots.append((float(x), float(y), 0.5 * m.size))
    return spots

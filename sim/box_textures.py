"""Procedural appearance of the cardboard boxes that flow down the inspection line.

A box is described by a :class:`BoxSpec` (size, colour, defects, distractor marks). From it we
render six face textures: kraft-cardboard base, packing tape, printed marks, distractor stickers
(FRAGILE, shipping label, QC PASS, ...) and the defects the VLM has to find (NG / REJECT labels,
tears, dents). Geometric parts of defects (dent / tear depressions) live in the spec as ``depth``
and are applied to the mesh by ``sim/boxes.py``.

This module only depends on numpy + Pillow so it can run both on the host and inside Isaac Sim.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_COND = FONT_DIR / "DejaVuSansCondensed-Bold.ttf"
FONT_REG = FONT_DIR / "DejaVuSans.ttf"
CJK_FONT_CANDIDATES = (
    FONT_DIR / "NotoSansCJK-Bold.ttc",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
)

# Face frames in box-local coordinates: (normal, u axis, v axis) with u x v = normal.
# Side faces have v = +z so printed text is upright when the box stands upright.
FACE_FRAMES = {
    "px": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "nx": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "py": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "ny": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "pz": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "nz": ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
}
FACES = tuple(FACE_FRAMES)
SIDE_FACES = ("px", "nx", "py", "ny")

DEFECT_KINDS = ("ng_label", "reject_label", "tear", "dent")
# Where each defect may appear: always in view of the inspection station. Printed marks go on the top face
# (overhead camera); damage may also be on the +y side, which faces the station's side camera.
DEFECT_FACES = {"ng_label": ("pz",), "reject_label": ("pz",), "tear": ("pz", "py"), "dent": ("pz", "py")}
DISTRACTOR_KINDS = ("fragile", "shipping_label", "qc_pass", "this_side_up")

KRAFT = (178, 137, 94)
TAPE_WIDTH = 0.048  # m


def face_extent(size: tuple[float, float, float], face: str) -> tuple[float, float]:
    """(width along u, height along v) of a face in metres."""
    _, u, v = FACE_FRAMES[face]
    return size[int(np.argmax(np.abs(u)))], size[int(np.argmax(np.abs(v)))]


def face_point(size: tuple[float, float, float], face: str, uv: tuple[float, float]) -> np.ndarray:
    """Box-local 3D point of a normalised (u, v) face coordinate."""
    n, u, v = (np.asarray(a, float) for a in FACE_FRAMES[face])
    half = np.asarray(size, float) / 2.0
    fu, fv = face_extent(size, face)
    return (np.abs(n) @ half) * n + (uv[0] - 0.5) * fu * u + (uv[1] - 0.5) * fv * v


@dataclass
class Mark:
    """Anything drawn on (or pressed into) a face."""

    kind: str
    face: str
    uv: tuple[float, float]  # centre, normalised face coords (v points up)
    size: float  # nominal width in metres (diameter for dents)
    rot: float = 0.0  # degrees, counter-clockwise in the image
    variant: str = ""
    depth: float = 0.0  # inward geometric depression in metres (dents / tears)


@dataclass
class BoxSpec:
    box_id: str
    size: tuple[float, float, float]  # (L along x, W along y, H along z) in metres
    tint: tuple[int, int, int]
    seed: int
    defects: list[Mark] = field(default_factory=list)
    distractors: list[Mark] = field(default_factory=list)

    @property
    def is_defective(self) -> bool:
        return bool(self.defects)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_defective"] = self.is_defective
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BoxSpec":
        return cls(
            box_id=d["box_id"],
            size=tuple(d["size"]),
            tint=tuple(d["tint"]),
            seed=d["seed"],
            defects=[Mark(**m) for m in d["defects"]],
            distractors=[Mark(**m) for m in d["distractors"]],
        )


# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------

# (min width, max width, height / width) of each mark's footprint in metres.
_FOOTPRINT = {
    "ng_label": (0.05, 0.08, 1.0),
    "reject_label": (0.10, 0.14, 0.42),
    "tear": (0.06, 0.11, 0.6),
    "dent": (0.07, 0.13, 1.0),
    "fragile": (0.09, 0.12, 0.5),
    "shipping_label": (0.09, 0.11, 0.7),
    "qc_pass": (0.04, 0.055, 1.0),
    "this_side_up": (0.07, 0.10, 0.75),
}
_VARIANTS = {
    "ng_label": ("round", "square", "marker", "jp_stamp"),
    "reject_label": ("stamp", "sticker"),
    "tear": ("liner", "puncture"),
    "dent": ("dent", "crushed_edge"),
    "fragile": ("white", "red"),
}
_VARIANT_P = {"ng_label": (0.35, 0.3, 0.25, 0.1)}


def _cjk_font() -> Path | None:
    return next((p for p in CJK_FONT_CANDIDATES if p.exists()), None)


def sample_box_spec(
    rng: np.random.Generator,
    box_id: str,
    p_defect: float = 0.5,
    defect_kinds: tuple[str, ...] = DEFECT_KINDS,
    defect_faces: dict[str, tuple[str, ...]] = DEFECT_FACES,
    max_distractors: int = 3,
) -> BoxSpec:
    """Random box. ``defect_faces`` maps each defect kind to the faces it may appear on."""
    length = rng.uniform(0.24, 0.36)
    width = rng.uniform(0.18, min(0.28, length))
    height = rng.uniform(0.12, 0.20)
    size = (round(length, 3), round(width, 3), round(height, 3))
    # Brightness varies a lot between suppliers, hue only slightly (keeps it reading as kraft).
    tint = tuple(int(np.clip(c * rng.uniform(0.86, 1.08) + rng.normal(0, 3), 0, 255)) for c in KRAFT)
    spec = BoxSpec(box_id=box_id, size=size, tint=tint, seed=int(rng.integers(2**31)))

    placed: list[Mark] = []
    if rng.random() < p_defect:
        n_defects = min(1 if rng.random() < 0.8 else 2, len(defect_kinds))
        for kind in rng.choice(defect_kinds, size=n_defects, replace=False):
            mark = _place(rng, spec, str(kind), defect_faces[str(kind)], placed)
            if mark:
                spec.defects.append(mark)
                placed.append(mark)

    n_distractors = int(rng.integers(1, max_distractors + 1))
    for kind in rng.choice(DISTRACTOR_KINDS, size=n_distractors, replace=False):
        faces = SIDE_FACES if kind == "this_side_up" else ("pz", *SIDE_FACES)
        mark = _place(rng, spec, str(kind), faces, placed)
        if mark:
            spec.distractors.append(mark)
            placed.append(mark)
    return spec


def _place(rng, spec: BoxSpec, kind: str, faces, placed: list[Mark]) -> Mark | None:
    wmin, wmax, aspect = _FOOTPRINT[kind]
    variants = _VARIANTS.get(kind, ("",))
    p = _VARIANT_P.get(kind)
    if kind == "ng_label" and _cjk_font() is None:
        variants, p = variants[:-1], None
    variant = str(rng.choice(variants, p=p)) if p else str(rng.choice(variants))

    # Pick the face first and retry positions on it (the tape band rejects many spots on the top face, so
    # re-drawing the face on every retry would push marks onto the sides).
    order = [str(f) for f in rng.permutation(faces)]
    for attempt in range(60):
        face = order[min(attempt // 20, len(order) - 1)]
        fu, fv = face_extent(spec.size, face)
        width = min(rng.uniform(wmin, wmax), 0.75 * fu, 0.75 * fv / max(aspect, 1e-3))
        height = width * aspect
        rot = 0.0
        if kind in ("ng_label", "reject_label", "fragile", "shipping_label", "qc_pass"):
            rot = rng.normal(0, 6) if variant != "stamp" else rng.uniform(-25, 25)
        elif kind == "tear":
            rot = rng.uniform(0, 180)
        r = 0.5 * math.hypot(width, height)
        margin = 0.012
        if kind == "dent":
            # Dents (especially crushed edges) may sit right at the face border.
            mu = mv = -0.25 * width if variant == "crushed_edge" else 0.2 * width
        else:
            mu, mv = r + margin, r + margin
        if fu - 2 * mu <= 0 or fv - 2 * mv <= 0:
            continue
        u = rng.uniform(mu, fu - mu) / fu
        v = rng.uniform(mv, fv - mv) / fv
        if variant == "crushed_edge":
            # Push the centre onto a random face edge.
            if rng.random() < 0.5:
                u = float(rng.choice([0.04, 0.96]))
            else:
                v = float(rng.choice([0.04, 0.96]))
        crosses_tape = face in ("pz", "nz") and abs(v - 0.5) * fv < TAPE_WIDTH / 2 + 0.5 * height + 0.005
        if crosses_tape and kind not in ("ng_label", "reject_label", "dent"):
            continue  # keep printed marks and tears off the packing tape; reject labels get slapped across it
        if any(
            m.face == face
            and math.hypot((m.uv[0] - u) * fu, (m.uv[1] - v) * fv) < r + 0.5 * math.hypot(*_mark_wh(m)) + 0.01
            for m in placed
        ):
            continue
        depth = 0.0
        if kind == "dent":
            depth = rng.uniform(0.015, 0.03) if variant == "dent" else rng.uniform(0.025, 0.045)
        elif kind == "tear":
            depth = rng.uniform(0.003, 0.006) if variant == "liner" else rng.uniform(0.01, 0.018)
        return Mark(kind, face, (round(u, 4), round(v, 4)), round(width, 4), round(float(rot), 2), variant,
                    round(depth, 4))
    return None


def _mark_wh(m: Mark) -> tuple[float, float]:
    return m.size, m.size * _FOOTPRINT[m.kind][2]


# --------------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------------


def _smooth_noise(h: int, w: int, cell: float, rng) -> np.ndarray:
    """Value noise in roughly [-1, 1] with a feature size of ``cell`` pixels."""
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    small = rng.uniform(0, 1, (gh, gw)).astype(np.float32)
    img = Image.fromarray(small, "F").resize((w, h), Image.BICUBIC)
    return np.clip(np.asarray(img, np.float32) * 2.0 - 1.0, -1.5, 1.5)


def _font(path: Path, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), max(4, int(px)))


def _fit_text(draw: ImageDraw.ImageDraw, text: str, path: Path, max_w: float, max_h: float):
    size = int(max_h)
    while size > 4:
        font = _font(path, size)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        if x1 - x0 <= max_w and y1 - y0 <= max_h:
            return font
        size = int(size * 0.92)
    return _font(path, 4)


def _draw_centered(draw, text, font, cx, cy, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (x0 + x1) / 2, cy - (y0 + y1) / 2), text, font=font, fill=fill)


def _warp(img: Image.Image, amp: float, cell: float, rng) -> Image.Image:
    """Smooth random displacement (hand-drawn / stamped wobble)."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    dx = _smooth_noise(h, w, cell, rng) * amp
    dy = _smooth_noise(h, w, cell, rng) * amp
    yy, xx = np.mgrid[0:h, 0:w]
    sx = np.clip((xx + dx).round().astype(int), 0, w - 1)
    sy = np.clip((yy + dy).round().astype(int), 0, h - 1)
    return Image.fromarray(a[sy, sx])


def _ink_speckle(patch: Image.Image, rng, coverage: float = 0.85) -> Image.Image:
    """Uneven ink: knock random holes into the alpha channel (rubber stamp look)."""
    a = np.asarray(patch).astype(np.float32)
    h, w = a.shape[:2]
    blotch = _smooth_noise(h, w, max(3.0, w / 18), rng)
    fine = rng.uniform(0, 1, (h, w))
    keep = np.clip((blotch * 0.5 + 0.5) * 0.6 + fine * 0.6 + (coverage - 0.6), 0, 1)
    a[..., 3] *= np.where(keep > 0.55, 1.0, keep * 0.6)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def _with_shadow(patch: Image.Image, px: int) -> Image.Image:
    """Add a soft drop shadow under a sticker (makes it read as a physical label)."""
    pad = 2 * px + 2
    out = Image.new("RGBA", (patch.width + 2 * pad, patch.height + 2 * pad), (0, 0, 0, 0))
    alpha = patch.split()[3]
    shadow = Image.new("RGBA", patch.size, (20, 12, 5, 0))
    shadow.putalpha(alpha.point(lambda x: int(x * 0.35)))
    out.alpha_composite(shadow, (pad + px, pad + px))
    out = out.filter(ImageFilter.GaussianBlur(px))
    out.alpha_composite(patch, (pad, pad))
    return out


# --------------------------------------------------------------------------------------------
# Mark patches (RGBA, unrotated, width = ``w`` px)
# --------------------------------------------------------------------------------------------

RED = (206, 28, 36)
INK_RED = (190, 30, 40)
WHITE = (246, 244, 238)


def _patch_ng_label(w: int, variant: str, rng) -> Image.Image:
    h = w
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if variant == "round":
        d.ellipse((0, 0, w - 1, h - 1), fill=RED + (255,))
        d.ellipse((w * 0.06, h * 0.06, w * 0.94, h * 0.94), outline=WHITE + (255,), width=max(1, w // 40))
        _draw_centered(d, "NG", _fit_text(d, "NG", FONT_BOLD, w * 0.62, h * 0.5), w / 2, h / 2, WHITE + (255,))
        return _with_shadow(img, max(1, w // 60))
    if variant == "square":
        bw = max(2, w // 11)
        d.rectangle((0, 0, w - 1, h - 1), fill=WHITE + (255,))
        d.rectangle((bw / 2, bw / 2, w - 1 - bw / 2, h - 1 - bw / 2), outline=RED + (255,), width=bw)
        _draw_centered(d, "NG", _fit_text(d, "NG", FONT_BOLD, w * 0.66, h * 0.55), w / 2, h / 2, RED + (255,))
        return _with_shadow(img, max(1, w // 60))
    if variant == "jp_stamp":
        cjk = _cjk_font()
        bw = max(2, w // 16)
        d.ellipse((bw, bw, w - 1 - bw, h - 1 - bw), outline=INK_RED + (235,), width=bw)
        _draw_centered(d, "不良", _fit_text(d, "不良", cjk, w * 0.66, h * 0.4), w / 2, h / 2, INK_RED + (235,))
        return _ink_speckle(_warp(img, w / 120, w / 6, rng), rng)
    # marker: hand-written "NG" in red or black marker
    color = (INK_RED if rng.random() < 0.6 else (25, 22, 24)) + (235,)
    big = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    font = _fit_text(bd, "NG", FONT_REG, w * 1.7, h * 1.2)
    _draw_centered(bd, "NG", font, w, h, color)
    big = big.filter(ImageFilter.MaxFilter(max(3, (w // 22) | 1)))
    big = _warp(big, w / 14, w / 3, rng)
    if rng.random() < 0.5:  # circled
        bd = ImageDraw.Draw(big)
        bd.ellipse((w * 0.08, h * 0.25, w * 1.92, h * 1.75), outline=color, width=max(2, w // 18))
        big = _warp(big, w / 30, w / 2, rng)
    return big.resize((w, h), Image.LANCZOS)


def _patch_reject_label(w: int, variant: str, rng) -> Image.Image:
    h = int(w * _FOOTPRINT["reject_label"][2])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    text = "REJECT" if rng.random() < 0.6 else "REJECTED"
    if variant == "sticker":
        d.rectangle((0, 0, w - 1, h - 1), fill=RED + (255,))
        d.rectangle((w * 0.03, h * 0.08, w * 0.97, h * 0.92), outline=WHITE + (255,), width=max(1, w // 70))
        _draw_centered(d, text, _fit_text(d, text, FONT_BOLD, w * 0.84, h * 0.58), w / 2, h / 2, WHITE + (255,))
        return _with_shadow(img, max(1, w // 80))
    bw = max(2, w // 45)
    col = INK_RED + (230,)
    d.rectangle((bw, bw, w - 1 - bw, h - 1 - bw), outline=col, width=bw)
    d.rectangle((3.2 * bw, 3.2 * bw, w - 1 - 3.2 * bw, h - 1 - 3.2 * bw), outline=col, width=max(1, bw // 2))
    _draw_centered(d, text, _fit_text(d, text, FONT_COND, w * 0.8, h * 0.52), w / 2, h / 2, col)
    return _ink_speckle(_warp(img, w / 220, w / 8, rng), rng)


def _patch_fragile(w: int, variant: str, rng) -> Image.Image:
    h = int(w * _FOOTPRINT["fragile"][2])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bg, fg = (RED, WHITE) if variant == "red" else (WHITE, RED)
    d.rectangle((0, 0, w - 1, h - 1), fill=bg + (255,))
    d.rectangle((w * 0.03, h * 0.06, w * 0.97, h * 0.94), outline=fg + (255,), width=max(1, w // 60))
    # wine-glass pictogram on the left
    gx, gy, gs = w * 0.15, h * 0.5, h * 0.62
    d.polygon([(gx - gs * 0.28, gy - gs * 0.5), (gx + gs * 0.28, gy - gs * 0.5), (gx, gy)], fill=fg + (255,))
    d.line([(gx, gy), (gx, gy + gs * 0.38)], fill=fg + (255,), width=max(1, w // 70))
    d.line([(gx - gs * 0.2, gy + gs * 0.42), (gx + gs * 0.2, gy + gs * 0.42)], fill=fg + (255,), width=max(1, w // 70))
    _draw_centered(d, "FRAGILE", _fit_text(d, "FRAGILE", FONT_BOLD, w * 0.62, h * 0.42), w * 0.6, h * 0.4, fg + (255,))
    small = "HANDLE WITH CARE"
    _draw_centered(d, small, _fit_text(d, small, FONT_COND, w * 0.6, h * 0.16), w * 0.6, h * 0.74, fg + (255,))
    return _with_shadow(img, max(1, w // 80))


_TOKENS = ("LINE 3", "LINE 7", "BAY 12", "BAY 04", "ZONE C", "ZONE A", "DOCK 2", "CELL B-4", "WH-EAST",
           "ASSY 21", "STN 05", "RACK 18")


def _patch_shipping_label(w: int, rng) -> Image.Image:
    h = int(w * _FOOTPRINT["shipping_label"][2])
    img = Image.new("RGBA", (w, h), (250, 250, 247, 255))
    d = ImageDraw.Draw(img)
    ink = (20, 20, 22, 255)
    d.rectangle((0, 0, w - 1, h - 1), outline=(150, 150, 150, 255), width=1)
    d.text((w * 0.05, h * 0.04), "SHIP TO:", font=_font(FONT_BOLD, h * 0.09), fill=ink)
    for i in range(2):
        line = f"{rng.choice(_TOKENS)} / {rng.choice(_TOKENS)}"
        d.text((w * 0.05, h * (0.16 + 0.12 * i)), line, font=_font(FONT_REG, h * 0.085), fill=ink)
    lot = f"LOT {rng.integers(20, 27)}-{rng.integers(1000, 9999)}"
    d.text((w * 0.05, h * 0.42), lot, font=_font(FONT_BOLD, h * 0.12), fill=ink)
    d.line([(w * 0.04, h * 0.58), (w * 0.96, h * 0.58)], fill=ink, width=max(1, h // 80))
    x, x1 = w * 0.07, w * 0.93
    while x < x1:
        bw = rng.choice([1, 1, 2, 3]) * max(1.0, w / 160)
        if rng.random() < 0.55:
            d.rectangle((x, h * 0.64, min(x + bw, x1), h * 0.9), fill=ink)
        x += bw
    return _with_shadow(img, max(1, w // 90))


def _patch_qc_pass(w: int, rng) -> Image.Image:
    img = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    green = (34, 146, 72, 255)
    d.ellipse((0, 0, w - 1, w - 1), fill=green)
    d.ellipse((w * 0.07, w * 0.07, w * 0.93, w * 0.93), outline=WHITE + (255,), width=max(1, w // 30))
    _draw_centered(d, "QC", _fit_text(d, "QC", FONT_BOLD, w * 0.5, w * 0.34), w / 2, w * 0.42, WHITE + (255,))
    _draw_centered(d, "PASSED", _fit_text(d, "PASSED", FONT_COND, w * 0.62, w * 0.15), w / 2, w * 0.7,
                   WHITE + (255,))
    return _with_shadow(img, max(1, w // 60))


def _patch_this_side_up(w: int, rng) -> Image.Image:
    """Printed (ink) handling mark: two arrows over a bar + text."""
    h = int(w * _FOOTPRINT["this_side_up"][2])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink = (38, 30, 26, 225)
    for cx in (w * 0.33, w * 0.67):
        d.polygon([(cx - w * 0.12, h * 0.32), (cx + w * 0.12, h * 0.32), (cx, h * 0.04)], fill=ink)
        d.rectangle((cx - w * 0.045, h * 0.3, cx + w * 0.045, h * 0.62), fill=ink)
    d.rectangle((w * 0.12, h * 0.64, w * 0.88, h * 0.69), fill=ink)
    _draw_centered(d, "THIS SIDE UP", _fit_text(d, "THIS SIDE UP", FONT_COND, w * 0.9, h * 0.2), w / 2, h * 0.84,
                   ink)
    return _ink_speckle(img, rng, coverage=0.95)


def _paste(canvas: Image.Image, patch: Image.Image, cx: float, cy: float, rot: float) -> None:
    if rot:
        patch = patch.rotate(rot, resample=Image.BICUBIC, expand=True)
    canvas.alpha_composite(patch, (int(round(cx - patch.width / 2)), int(round(cy - patch.height / 2))))


# --------------------------------------------------------------------------------------------
# Damage (drawn straight into the face pixels)
# --------------------------------------------------------------------------------------------


def _jagged_polygon(rng, cx, cy, a, b, rot_deg, n=72, rough=0.22):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    low = np.interp(th, np.linspace(0, 2 * np.pi, 9), np.r_[rng.uniform(-1, 1, 8), 0])
    low[-1] = low[0]
    r = 1.0 + 0.18 * low + rough * rng.uniform(-0.5, 0.5, n)
    x, y = a * r * np.cos(th), b * r * np.sin(th)
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    return [(cx + c * xi - s * yi, cy - (s * xi + c * yi)) for xi, yi in zip(x, y)]


def _mask(shape, poly) -> np.ndarray:
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(poly, fill=255)
    return np.asarray(m, np.float32) / 255.0


def _draw_tear(px: np.ndarray, mark: Mark, cx, cy, ppm: float, rng) -> None:
    h, w = px.shape[:2]
    a = mark.size * ppm / 2
    b = a * _FOOTPRINT["tear"][2]
    seed_state = rng.bit_generator.state
    outer = _mask((h, w), _jagged_polygon(rng, cx, cy, a, b, mark.rot, rough=0.25))
    rng.bit_generator.state = seed_state  # inner edge follows the outer one, then gets its own jaggedness
    inner = _mask((h, w), _jagged_polygon(rng, cx, cy, a * 0.8, b * 0.72, mark.rot, rough=0.35))
    rim = np.clip(outer - inner, 0, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    # Exposed corrugated medium: flutes run along v on side faces.
    period = 0.0045 * ppm
    flutes = 0.5 + 0.5 * np.sin(2 * np.pi * xx / max(period, 2.0))
    medium = np.stack([150 + 35 * flutes, 110 + 28 * flutes, 68 + 20 * flutes], -1)
    # Self-shadow along the upper edge of the opening.
    shadow = np.clip(inner - np.roll(inner, int(max(2, 0.004 * ppm)), axis=0), 0, 1)
    medium *= (1.0 - 0.45 * shadow)[..., None]
    fibres = 220 + 25 * rng.uniform(-1, 1, (h, w))
    rim_col = np.stack([fibres, fibres * 0.93, fibres * 0.8], -1)
    out = px * (1 - outer[..., None]) + rim_col * rim[..., None] + medium * inner[..., None]
    if mark.variant == "puncture":
        hole = _mask((h, w), _jagged_polygon(rng, cx, cy, a * 0.42, b * 0.4, mark.rot, rough=0.4))
        out = out * (1 - hole[..., None]) + np.array([38, 28, 20]) * hole[..., None]
    px[:] = out


def _draw_dent_creases(px: np.ndarray, mark: Mark, cx, cy, ppm: float, rng) -> None:
    h, w = px.shape[:2]
    rad = mark.size * ppm / 2
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad, 1) ** 2
    px *= (1 - 0.12 * np.exp(-d2 * 2.5))[..., None]  # scuffed / compressed paper is darker
    dark = Image.new("L", (w, h), 0)
    light = Image.new("L", (w, h), 0)
    dd, ld = ImageDraw.Draw(dark), ImageDraw.Draw(light)
    lw = max(1, int(0.0022 * ppm))
    for _ in range(int(rng.integers(2, 5))):
        ang = rng.uniform(0, np.pi)
        length = rad * rng.uniform(0.5, 1.1)
        off = rng.uniform(-0.3, 0.3) * rad
        bend = rng.uniform(-0.2, 0.2) * rad
        t = np.linspace(-0.5, 0.5, 12)
        pts_x = cx + off * -math.sin(ang) + t * length * math.cos(ang) + bend * (1 - 4 * t**2) * -math.sin(ang)
        pts_y = cy + off * math.cos(ang) + t * length * math.sin(ang) + bend * (1 - 4 * t**2) * math.cos(ang)
        pts = list(zip(pts_x, pts_y))
        dd.line(pts, fill=150, width=lw)
        ld.line([(x + lw, y + lw) for x, y in pts], fill=110, width=lw)
    blur = ImageFilter.GaussianBlur(1.2 * lw)
    dark_a = np.asarray(dark.filter(blur), np.float32)[..., None] / 255
    light_a = np.asarray(light.filter(blur), np.float32)[..., None] / 255
    # Folds are shading on the same paper: darken / brighten multiplicatively so the hue stays kraft.
    px *= (1 - 0.45 * dark_a) * (1 + 0.35 * light_a)


# --------------------------------------------------------------------------------------------
# Face rendering
# --------------------------------------------------------------------------------------------


def _cardboard(w: int, h: int, tint, ppm: float, rng) -> np.ndarray:
    base = np.asarray(tint, np.float32)
    shade = _smooth_noise(h, w, 0.08 * ppm, rng) * 0.045 + _smooth_noise(h, w, 0.015 * ppm, rng) * 0.02
    grain = rng.normal(0, 0.016, (h, w))
    streak = np.convolve(rng.normal(0, 1, w), np.ones(3) / 3, mode="same")[None, :] * 0.012
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)) / max(0.006 * ppm, 1)
    edge_dark = 0.09 * np.exp(-edge)
    f = 1.0 + shade + grain + streak - edge_dark
    return base[None, None, :] * f[..., None]


def _draw_tape(img: Image.Image, box: tuple[float, float, float, float], along_u: bool, rng) -> None:
    x0, y0, x1, y1 = box
    w, h = int(x1 - x0), int(y1 - y0)
    if w <= 0 or h <= 0:
        return
    tape = np.zeros((h, w, 4), np.float32)
    tape[..., :3] = (208, 172, 118)
    gloss = _smooth_noise(h, w, max(w, h) / 6, rng)
    gloss = np.repeat(gloss.mean(axis=1, keepdims=True), w, 1) if along_u else gloss
    tape[..., :3] *= (1 + 0.08 * gloss)[..., None]
    tape[..., 3] = 150
    edge = max(1, min(w, h) // 40)
    if along_u:
        tape[:edge, :, :3] *= 0.8
        tape[-edge:, :, :3] *= 0.8
    else:
        tape[:, :edge, :3] *= 0.8
        tape[:, -edge:, :3] *= 0.8
    img.alpha_composite(Image.fromarray(tape.clip(0, 255).astype(np.uint8)), (int(x0), int(y0)))


def render_face(spec: BoxSpec, face: str, px_per_m: float = 1800.0, max_px: int = 1024) -> Image.Image:
    fu, fv = face_extent(spec.size, face)
    ppm = min(px_per_m, max_px / max(fu, fv))
    w, h = max(8, int(round(fu * ppm))), max(8, int(round(fv * ppm)))
    rng = np.random.default_rng([spec.seed, FACES.index(face)])

    def to_px(uv):
        return uv[0] * w, (1.0 - uv[1]) * h

    pix = _cardboard(w, h, spec.tint, ppm, rng)
    marks = [m for m in spec.defects + spec.distractors if m.face == face]
    for m in marks:
        if m.kind == "dent":
            _draw_dent_creases(pix, m, *to_px(m.uv), ppm, rng)
    for m in marks:
        if m.kind == "tear":
            _draw_tear(pix, m, *to_px(m.uv), ppm, rng)
    img = Image.fromarray(pix.clip(0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img)

    tape_px = TAPE_WIDTH * ppm
    if face in ("pz", "nz"):  # flap seam + tape along x
        d.line([(0, h / 2), (w, h / 2)], fill=(70, 50, 32, 200), width=max(1, int(0.002 * ppm)))
        _draw_tape(img, (0, h / 2 - tape_px / 2, w, h / 2 + tape_px / 2), True, rng)
    elif face in ("px", "nx"):  # tape wraps over the ends
        tail = 0.06 * ppm
        for y_start in (0, h - tail):
            _draw_tape(img, (w / 2 - tape_px / 2, y_start, w / 2 + tape_px / 2, y_start + tail), False, rng)
    if face == "ny":  # printed part code on the front
        code = f"PART No. {rng.choice(['PX', 'MX', 'KT'])}-{rng.integers(1000, 9999)}   QTY {rng.integers(6, 48)}"
        font = _fit_text(d, code, FONT_COND, w * 0.6, 0.016 * ppm)
        d.text((w * 0.06, h * 0.08), code, font=font, fill=(45, 34, 28, 210))

    for m in marks:
        cx, cy = to_px(m.uv)
        mw = max(8, int(m.size * ppm))
        if m.kind == "ng_label":
            patch = _patch_ng_label(mw, m.variant, rng)
        elif m.kind == "reject_label":
            patch = _patch_reject_label(mw, m.variant, rng)
        elif m.kind == "fragile":
            patch = _patch_fragile(mw, m.variant, rng)
        elif m.kind == "shipping_label":
            patch = _patch_shipping_label(mw, rng)
        elif m.kind == "qc_pass":
            patch = _patch_qc_pass(mw, rng)
        elif m.kind == "this_side_up":
            patch = _patch_this_side_up(mw, rng)
        else:
            continue
        _paste(img, patch, cx, cy, m.rot)
    return img.convert("RGB")


def render_box(spec: BoxSpec, out_dir: Path | str, px_per_m: float = 1800.0) -> dict[str, Path]:
    """Write ``<out_dir>/<box_id>/{face}.png`` + ``spec.json``; returns face -> texture path."""
    box_dir = Path(out_dir) / spec.box_id
    box_dir.mkdir(parents=True, exist_ok=True)
    paths = {face: box_dir / f"{face}.png" for face in FACES}
    spec_path = box_dir / "spec.json"
    payload = json.dumps(spec.to_dict(), indent=2)
    if spec_path.exists() and spec_path.read_text() == payload and all(p.exists() for p in paths.values()):
        return paths  # already rendered (textures are deterministic given the spec)
    for face in FACES:
        render_face(spec, face, px_per_m).save(paths[face])
    spec_path.write_text(payload)
    return paths


def render_net(spec: BoxSpec, px_per_m: float = 1200.0) -> Image.Image:
    """Unfolded box (cross layout) for quick visual checks:  top row pz, middle nx|ny|px|py, bottom nz."""
    faces = {f: render_face(spec, f, px_per_m, max_px=10_000) for f in FACES}
    L, W, H = (int(round(s * px_per_m)) for s in spec.size)
    net = Image.new("RGB", (2 * L + 2 * W + 5, 2 * W + H + 4), (255, 255, 255))
    for face, xy in {"pz": (W + 1, 0), "nx": (0, W + 2), "ny": (W + 1, W + 2), "px": (W + L + 2, W + 2),
                     "py": (2 * W + L + 3, W + 2), "nz": (W + 1, W + H + 3)}.items():
        net.paste(faces[face].resize(
            (W if face in ("px", "nx") else L, H if face in SIDE_FACES else W)), xy)
    return net

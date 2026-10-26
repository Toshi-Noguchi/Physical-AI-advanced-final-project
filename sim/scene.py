"""Factory inspection cell: conveyor line with an inspection station, an NG (reject) station, a mobile
manipulator (Ridgeback + Franka with a suction tool) and cameras.

Everything here is static USD authoring; per-step behaviour (conveyor control, box spawning, camera
readout) lives in :mod:`env`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from box_textures import FONT_BOLD, _cjk_font, _fit_text
from boxes import physics_material, textured_material
from PIL import Image, ImageDraw
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

ASSETS = {
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "conveyor": "/Isaac/Props/Conveyors/ConveyorBelt_A08.usd",
    "robot": "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
    "suction": "/Isaac/Robots/UniversalRobots/ur10/grippers/short_gripper.usd",
}
CONVEYOR_LENGTH = 2.7187  # m, ConveyorBelt_A08 spans x in [-L, 0] in its own frame

ROBOT_PATH = "/World/Robot"
HAND_PATH = f"{ROBOT_PATH}/panda_hand"
GRIPPER_PATH = "/World/SuctionGripper"
TOOL_TIP_Z = 0.161  # suction cup face, metres along panda_hand +z (panda_hand origin = arm flange)
BASE_HALF_LENGTH = 0.48  # Ridgeback footprint (x), metres
# DROID's reset pose: MolmoAct2-DROID was trained on episodes starting here (cup face ~0.36 m ahead of and
# ~0.33 m above panda_link0, pointing down).
ARM_HOME = (0.0, -math.pi / 5, 0.0, -4 * math.pi / 5, 0.0, 3 * math.pi / 5, 0.0)


@dataclass
class Layout:
    inspect_x: float = 0.0  # box centres stop here for inspection
    # The stock conveyor is 1.15 m wide with the belt at 0.77 m: too far / high for the Franka on the Ridgeback,
    # whose DROID home pose holds the cup only ~0.61 m above the floor. It is squeezed to a narrow line and sunk
    # into the floor so the belt runs at ~0.33 m and box tops stay below the cup.
    conveyor_y_scale: float = 0.5
    conveyor_z_scale: float = 0.7
    conveyor_sink: float = 0.21
    conveyor_segments: tuple[float, ...] = (-1.2, 1.5)  # downstream end x of each conveyor segment
    belt_speed: float = 0.3  # m/s
    spawn_x: float = -3.6
    despawn_x: float = 1.3
    ng_table_center: tuple[float, float] = (2.6, -3.2)
    ng_table_size: tuple[float, float, float] = (1.8, 1.3, 0.42)  # x, y, top height; the robot works its -y side
    # Filled in by build_scene from the asset bounds:
    belt_top_z: float = 0.0
    belt_half_width: float = 0.0
    frame_half_width: float = 0.0
    robot_home: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw of the base


# --------------------------------------------------------------------------------------------
# Small USD helpers
# --------------------------------------------------------------------------------------------


def quat_yaw(yaw: float) -> Gf.Quatf:
    return Gf.Quatf(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> Gf.Quatf:
    """Orientation of a USD camera (looks down -Z, +Y up) at ``eye`` looking at ``target``."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:  # looking along the up hint
        x = np.cross((0.0, 1.0, 0.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = Gf.Matrix3d(*x, *y, *z)  # rows = camera axes in world (USD row-vector convention)
    q = m.ExtractRotation().GetQuat()
    return Gf.Quatf(q.GetReal(), *q.GetImaginary())


_DOUBLE_TYPES = (Sdf.ValueTypeNames.Double3, Sdf.ValueTypeNames.Quatd)


def _add_op(xformable: UsdGeom.Xformable, kind: str, value) -> None:
    """Add a translate/orient/scale op, matching the precision of an op a referenced asset already has."""
    attr = xformable.GetPrim().GetAttribute(f"xformOp:{kind}")
    double = attr.IsValid() and attr.GetTypeName() in _DOUBLE_TYPES if attr.IsValid() else kind == "translate"
    precision = UsdGeom.XformOp.PrecisionDouble if double else UsdGeom.XformOp.PrecisionFloat
    op = {"translate": xformable.AddTranslateOp, "orient": xformable.AddOrientOp, "scale": xformable.AddScaleOp}[kind]
    if kind == "orient":
        value = Gf.Quatd(value.GetReal(), *value.GetImaginary()) if double else Gf.Quatf(value.GetReal(), *value.GetImaginary())
    else:
        value = Gf.Vec3d(*value) if double else Gf.Vec3f(*value)
    op(precision).Set(value)


def set_pose(xformable: UsdGeom.Xformable, pos, quat: Gf.Quatf | None = None, scale=None) -> None:
    xformable.ClearXformOpOrder()
    _add_op(xformable, "translate", pos)
    if quat is not None:
        _add_op(xformable, "orient", quat)
    if scale is not None:
        _add_op(xformable, "scale", scale)


_MATS: dict[str, UsdShade.Material] = {}


def color_material(stage, name: str, rgb, roughness=0.5, metallic=0.0, emissive=None) -> UsdShade.Material:
    path = f"/World/Looks/{name}"
    if path in _MATS and _MATS[path].GetPrim().IsValid():
        return _MATS[path]
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if emissive is not None:
        pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    _MATS[path] = mat
    return mat


def bind(prim, mat: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def cuboid(stage, path, center, size, mat=None, collide=False, yaw=0.0) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_pose(cube, center, quat_yaw(yaw) if yaw else None, size)
    if mat is not None:
        bind(cube.GetPrim(), mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def cylinder(stage, path, center, radius, height, mat=None, axis="Z", collide=False) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    set_pose(cyl, center)
    if mat is not None:
        bind(cyl.GetPrim(), mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def textured_quad(stage, path, center, size, quat: Gf.Quatf, texture: Path) -> UsdGeom.Mesh:
    """Flat textured rectangle in its local XY plane (normal +Z), e.g. signs and floor markings."""
    w, h = size
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([(-w / 2, -h / 2, 0), (w / 2, h / 2, 0)])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 1)]))
    set_pose(mesh, center, quat)
    bind(mesh.GetPrim(), textured_material(stage, f"{path}_mat", texture, roughness=0.6))
    return mesh


def world_range(stage, path: str) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()


# --------------------------------------------------------------------------------------------
# Sign textures
# --------------------------------------------------------------------------------------------


def make_sign(path: Path, lines: list[tuple[str, float]], bg, fg, size=(800, 560), border=True) -> Path:
    """Render a sign: ``lines`` = [(text, relative height)], stacked vertically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    w, h = size
    if border:
        d.rectangle((10, 10, w - 11, h - 11), outline=fg, width=12)
    total = sum(r for _, r in lines)
    y = h * 0.1
    for text, rel in lines:
        lh = h * 0.8 * rel / total
        font_path = _cjk_font() if any(ord(c) > 0x2000 for c in text) else FONT_BOLD
        if font_path is None:
            y += lh
            continue
        font = _fit_text(d, text, font_path, w * 0.86, lh * 0.8)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        d.text(((w - (x1 - x0)) / 2 - x0, y + (lh - (y1 - y0)) / 2 - y0), text, font=font, fill=fg)
        y += lh
    img.save(path)
    return path


def make_hatch(path: Path, size=(1024, 64), colors=((200, 30, 30), (235, 235, 235)), stripe=48) -> Path:
    """Diagonal two-colour floor tape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    band = ((xx + yy) // stripe) % 2
    img = np.where(band[..., None] == 0, np.array(colors[0]), np.array(colors[1])).astype(np.uint8)
    Image.fromarray(img).save(path)
    return path


# --------------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------------


def build_scene(stage, assets_root: str, layout: Layout, gen_dir: Path) -> Layout:
    """Author the whole cell on ``stage``. Returns ``layout`` with the measured fields filled in."""
    _MATS.clear()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    for name in ("Env", "Line", "NGStation", "Cameras", "Boxes", "Looks", "PhysicsMaterials"):
        UsdGeom.Scope.Define(stage, f"/World/{name}") if name in ("Looks", "PhysicsMaterials") else UsdGeom.Xform.Define(
            stage, f"/World/{name}"
        )
    physics_material(stage, "/World/PhysicsMaterials/cardboard", 0.7, 0.6)
    belt_mat = physics_material(stage, "/World/PhysicsMaterials/belt", 1.0, 0.9)

    _build_environment(stage, assets_root)
    _build_conveyor(stage, assets_root, layout, belt_mat)
    _build_inspection_station(stage, layout, gen_dir)
    _build_ng_station(stage, layout, gen_dir)
    _build_robot(stage, assets_root, layout)
    _build_cameras(stage, assets_root, layout)
    return layout


def _build_environment(stage, assets_root: str) -> None:
    env = UsdGeom.Xform.Define(stage, "/World/Env/warehouse")
    env.GetPrim().GetReferences().AddReference(assets_root + ASSETS["warehouse"])
    # Our own floor collider (the warehouse shell is visual first).
    floor = cuboid(stage, "/World/Env/floor_collider", (0, 0, -0.05), (40, 40, 0.1), collide=True)
    floor.MakeInvisible()

    dome = UsdLux.DomeLight.Define(stage, "/World/Env/dome")
    dome.CreateIntensityAttr(300.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Env/key")
    sun.CreateIntensityAttr(1000.0)
    sun.CreateAngleAttr(2.0)
    set_pose(sun, (0, 0, 10), Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, -1, 0), 40).GetQuat()))
    for i, x in enumerate((-2.0, 1.5)):  # ceiling panels over the cell
        panel = UsdLux.RectLight.Define(stage, f"/World/Env/ceiling_{i}")
        panel.CreateWidthAttr(2.0)
        panel.CreateHeightAttr(0.6)
        panel.CreateIntensityAttr(3500.0)
        set_pose(panel, (x, -0.8, 4.0))


def _build_conveyor(stage, assets_root: str, layout: Layout, belt_mat) -> None:
    for i, end_x in enumerate(layout.conveyor_segments):
        seg = UsdGeom.Xform.Define(stage, f"/World/Line/conveyor_{i}")
        seg.GetPrim().GetReferences().AddReference(assets_root + ASSETS["conveyor"])
        set_pose(seg, (end_x, 0, -layout.conveyor_sink), None, (1.0, layout.conveyor_y_scale, layout.conveyor_z_scale))
        # The belt below does the transport; keep the stock rollers out of the contact pairs.
        for prim in Usd.PrimRange(seg.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    rollers = world_range(stage, "/World/Line/conveyor_0/Rollers")
    frame = world_range(stage, "/World/Line/conveyor_0")
    layout.belt_top_z = float(rollers.GetMax()[2]) + 0.002
    layout.belt_half_width = float(rollers.GetMax()[1])
    layout.frame_half_width = float(frame.GetMax()[1])

    x0 = layout.conveyor_segments[0] - CONVEYOR_LENGTH
    x1 = layout.conveyor_segments[-1]
    # Kinematic belt body without scale (surface velocity is expressed in the body frame), collider below it.
    belt = UsdGeom.Xform.Define(stage, "/World/Line/belt")
    set_pose(belt, ((x0 + x1) / 2, 0, layout.belt_top_z - 0.01))
    collider = cuboid(stage, "/World/Line/belt/collider", (0, 0, 0), (x1 - x0, 2 * layout.belt_half_width, 0.02),
                      collide=True)
    collider.MakeInvisible()
    prim = belt.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(belt_mat, UsdShade.Tokens.weakerThanDescendants,
                                                               "physics")
    sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
    sv.CreateSurfaceVelocityEnabledAttr(True)
    sv.CreateSurfaceVelocityAttr(Gf.Vec3f(0, 0, 0))  # env.py drives it

    # Low side guides so boxes stay on the rollers.
    guide_mat = color_material(stage, "guide_yellow", (0.85, 0.65, 0.05), 0.4, 0.3)
    for side, y in (("l", 1), ("r", -1)):
        cuboid(
            stage,
            f"/World/Line/guide_{side}",
            ((x0 + x1) / 2, y * (layout.belt_half_width + 0.01), layout.belt_top_z + 0.03),
            (x1 - x0, 0.02, 0.05),
            guide_mat,
            collide=True,
        )


def _build_inspection_station(stage, layout: Layout, gen_dir: Path) -> None:
    """Camera gantry on the far (+y) side of the line, plus signage and a stop light."""
    ix, top = layout.inspect_x, layout.belt_top_z
    alu = color_material(stage, "aluminium", (0.75, 0.76, 0.78), 0.35, 0.8)
    post_y = layout.frame_half_width + 0.15
    cam_z = top + 1.15
    sign_size = (0.45, 0.21)
    sign_z = cam_z + 0.13 + 0.03 + sign_size[1] / 2  # above the gantry arm (top face at cam_z + 0.13)
    post_h = sign_z + sign_size[1] / 2 + 0.02
    cuboid(stage, "/World/Line/inspect_post", (ix, post_y, post_h / 2), (0.06, 0.06, post_h), alu, collide=True)
    cuboid(stage, "/World/Line/inspect_arm", (ix, post_y / 2, cam_z + 0.1), (0.06, post_y + 0.06, 0.06), alu)
    led = UsdLux.RectLight.Define(stage, "/World/Line/inspect_led")
    led.CreateWidthAttr(0.5)
    led.CreateHeightAttr(0.08)
    led.CreateIntensityAttr(2500.0)
    set_pose(led, (ix, 0.12, cam_z + 0.05))  # RectLight emits along -Z: straight down onto the station
    _build_raking_lights(stage, layout)
    sign = make_sign(gen_dir / "sign_inspection.png", [("INSPECTION", 1.0), ("外観検査", 0.8)], (25, 70, 150),
                     (245, 245, 245), size=(900, 420))
    textured_quad(stage, "/World/Line/inspect_sign", (ix, post_y - 0.035, sign_z), sign_size,
                  Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat()), sign)  # faces the robot side (-y)
    # Stack light (green/amber/red) on the post.
    for k, (name, rgb) in enumerate((("green", (0.1, 0.8, 0.2)), ("amber", (1.0, 0.6, 0.0)), ("red", (0.9, 0.1, 0.1)))):
        cylinder(stage, f"/World/Line/stack_light_{name}", (ix + 0.07, post_y, 1.35 + 0.06 * k), 0.03, 0.055,
                 color_material(stage, f"lamp_{name}", rgb, 0.3, 0.0, emissive=tuple(0.2 * c for c in rgb)))


RAKING_LIGHTS = ("/World/Line/inspect_raking_up", "/World/Line/inspect_raking_down")
RAKING_INTENSITY = 400000.0  # while on; env.py keeps them off except for the inspection shot


def _build_raking_lights(stage, layout: Layout) -> None:
    """Low light bars up- and downstream on the far (+y) side of the belt. They light the box top (and its +y side)
    at a grazing angle, so dents and tears cast shadows that flat overhead lighting washes out. They stand outside
    the overhead camera's view (+-0.86 m by +-0.65 m on the floor), which the grasp policy was trained on."""
    ix, top = layout.inspect_x, layout.belt_top_z
    alu = color_material(stage, "aluminium", (0.75, 0.76, 0.78), 0.35, 0.8)
    dark = color_material(stage, "dark_metal", (0.08, 0.08, 0.09), 0.4, 0.6)
    height = top + 0.3  # just above the tallest box (0.20 m)
    for path, side in zip(RAKING_LIGHTS, (-1, 1)):
        eye = (ix + side * 0.7, 0.7, height)
        light = UsdLux.RectLight.Define(stage, path)
        light.CreateWidthAttr(0.3)
        light.CreateHeightAttr(0.06)
        light.CreateIntensityAttr(0.0)
        set_pose(light, eye, look_at_quat(eye, (ix, 0.0, top + 0.12)))
        # Housing just behind the emitting face (the light shines along its -Z), on a post from the floor.
        cuboid(stage, f"{path}/housing", (0.0, 0.0, 0.018), (0.32, 0.08, 0.03), dark)
        cylinder(stage, f"{path}_post", (eye[0], eye[1], (height - 0.03) / 2), 0.015, height - 0.03, alu)


def _build_ng_station(stage, layout: Layout, gen_dir: Path) -> None:
    cx, cy = layout.ng_table_center
    sx, sy, h = layout.ng_table_size
    steel = color_material(stage, "steel_gray", (0.45, 0.47, 0.5), 0.45, 0.7)
    red = color_material(stage, "ng_red", (0.75, 0.08, 0.08), 0.5, 0.1)
    cuboid(stage, "/World/NGStation/top", (cx, cy, h - 0.015), (sx, sy, 0.03), steel, collide=True)
    for i, (dx, dy) in enumerate((ux, uy) for ux in (-1, 0, 1) for uy in (-1, 1)):  # 6 legs for the long top
        cuboid(stage, f"/World/NGStation/leg_{i}", (cx + dx * (sx / 2 - 0.03), cy + dy * (sy / 2 - 0.03), (h - 0.03) / 2),
               (0.04, 0.04, h - 0.03), steel, collide=True)
    cuboid(stage, "/World/NGStation/shelf", (cx, cy, 0.12), (sx - 0.04, sy - 0.04, 0.02), steel)
    # Red lip around the top so the station reads as "reject" from any camera.
    for name, c, s in (("lip_n", (cx, cy + sy / 2, h + 0.02), (sx, 0.02, 0.04)),
                       ("lip_s", (cx, cy - sy / 2, h + 0.02), (sx, 0.02, 0.04)),
                       ("lip_e", (cx + sx / 2, cy, h + 0.02), (0.02, sy, 0.04)),
                       ("lip_w", (cx - sx / 2, cy, h + 0.02), (0.02, sy, 0.04))):
        cuboid(stage, f"/World/NGStation/{name}", c, s, red, collide=True)
    # Sign on a pole at the far (+x) end, facing the line and the approaching robot (-x).
    cuboid(stage, "/World/NGStation/sign_pole", (cx + sx / 2 + 0.08, cy, 0.8), (0.04, 0.04, 1.6), steel)
    sign = make_sign(gen_dir / "sign_ng.png", [("NG", 1.4), ("REJECT AREA", 0.5), ("不良品置き場", 0.6)],
                     (200, 25, 30), (250, 250, 250))
    textured_quad(stage, "/World/NGStation/sign", (cx + sx / 2 + 0.055, cy, 1.35), (0.5, 0.35),
                  quat_yaw(-math.pi / 2) * Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat()), sign)
    # Hatched floor tape around the station.
    hatch = make_hatch(gen_dir / "floor_hatch_red.png")
    m = 0.25
    for name, c, s, yaw in (("n", (cx, cy + sy / 2 + m, 0.002), (sx + 2 * m, 0.07), 0.0),
                            ("s", (cx, cy - sy / 2 - m, 0.002), (sx + 2 * m, 0.07), 0.0),
                            ("e", (cx + sx / 2 + m, cy, 0.002), (sy + 2 * m, 0.07), math.pi / 2),
                            ("w", (cx - sx / 2 - m, cy, 0.002), (sy + 2 * m, 0.07), math.pi / 2)):
        textured_quad(stage, f"/World/NGStation/tape_{name}", c, s, quat_yaw(yaw), hatch)


def _build_robot(stage, assets_root: str, layout: Layout) -> None:
    robot = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    robot.GetPrim().GetReferences().AddReference(assets_root + ASSETS["robot"])
    # Base faces the line (+y) from the near side; the arm reaches over the frame.
    layout.robot_home = (layout.inspect_x, -(layout.frame_half_width + 0.06 + BASE_HALF_LENGTH), math.pi / 2)

    # Replace the Franka Hand + fingers with a vacuum gripper: hide their meshes, drop their colliders.
    # (panda_hand's origin is the arm flange, so the gripper is authored as children of that link.)
    for link in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        for child in stage.GetPrimAtPath(f"{ROBOT_PATH}/{link}").GetChildren():
            if child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).MakeInvisible()
            for p in Usd.PrimRange(child):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
    _mount_suction_gripper(stage, assets_root)


# Attachment points of the surface gripper, (x, y) on the cup face in the hand frame.
CUP_OFFSETS = ((0.0, 0.0),)


def _mount_suction_gripper(stage, assets_root: str) -> None:
    """Mount Isaac's UR10 suction gripper (built along its +x) on the Franka flange (hand +z).

    Its meshes become part of the panda_hand link: the asset's own rigid body and gripper logic are
    removed and replaced by the surface gripper authored below, which is driven from env.py.
    """
    mount = UsdGeom.Xform.Define(stage, f"{HAND_PATH}/suction_mount")
    set_pose(mount, (0, 0, 0), Gf.Quatf(0.7071068, 0, -0.7071068, 0))  # -90 deg about y: asset +x -> hand +z
    # Reference on a child so the asset keeps its own root transform.
    g = UsdGeom.Xform.Define(stage, f"{HAND_PATH}/suction_mount/gripper")
    g.GetPrim().GetReferences().AddReference(assets_root + ASSETS["suction"])
    prim = g.GetPrim()
    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    prim.RemoveAPI(UsdPhysics.MassAPI)
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdPhysics.Joint) or p.GetTypeName() in ("IsaacSurfaceGripper", "DiskLight", "Camera"):
            p.SetActive(False)
        elif p.HasAPI(UsdPhysics.CollisionAPI):
            # Only the cup touches boxes; convex hulls because the hand is a dynamic link.
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(p.GetName() == "gripper_tip")
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

    # Surface gripper: attachment point(s) on the cup face, pointing along the hand's +z.
    from usd.schema.isaac import robot_schema  # available inside Isaac Sim only

    UsdGeom.Xform.Define(stage, GRIPPER_PATH)
    targets = []
    for i, (x, y) in enumerate(CUP_OFFSETS):
        attach = UsdPhysics.Joint.Define(stage, f"{GRIPPER_PATH}/attach_{i}")
        attach.CreateBody0Rel().SetTargets([Sdf.Path(HAND_PATH)])
        attach.CreateLocalPos0Attr().Set(Gf.Vec3f(x, y, TOOL_TIP_Z))
        attach.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        attach.CreateExcludeFromArticulationAttr().Set(True)
        for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(attach.GetPrim(), axis)
            limit.CreateLowAttr().Set(1.0)
            limit.CreateHighAttr().Set(-1.0)
        robot_schema.ApplyAttachmentPointAPI(attach.GetPrim())
        attach.GetPrim().CreateAttribute(robot_schema.Attributes.FORWARD_AXIS.name,
                                         robot_schema.Attributes.FORWARD_AXIS.type).Set("Z")
        attach.GetPrim().CreateAttribute(robot_schema.Attributes.CLEARANCE_OFFSET.name,
                                         robot_schema.Attributes.CLEARANCE_OFFSET.type).Set(0.005)
        targets.append(attach.GetPath())
    gripper = robot_schema.CreateSurfaceGripper(stage, f"{GRIPPER_PATH}/gripper")
    gripper.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets(targets)


CAMERAS = {
    # name: (resolution (h, w), focal length mm); horizontal aperture is 20.955 mm
    "overhead": ((960, 1280), 18.0),  # inspection (Molmo2) and grasping (MolmoAct2) view
    "side": ((720, 960), 16.0),  # inspection of the +y side
    "wrist": ((360, 640), 10.0),  # grasping view, ~90 deg like DROID's ZED mini
    "overview": ((720, 1280), 15.0),  # video only, not a camera in the cell
}
# Camera hardware shown where each in-cell camera is: asset and the lens it stands in for, in the asset's frame
# (all these models look along +x with +z up).
WRIST_CAM_OFFSET = 0.095  # m from the tool axis
# Direction of the camera from the tool axis in the panda_hand xy-plane: +x+y points away from the robot at the
# DROID home pose (joint 7 = 0), so the camera hangs over the workspace and the box stays mid-image.
WRIST_CAM_AZIMUTH = math.radians(45.0)
WRIST_CAM_TILT_DEG = 15.0  # toward the tool axis; the cup is ~41 deg off the camera position's vertical
SENSOR_MODELS = {
    "d455": ("/Isaac/Sensors/RealSense/D455/rsd455.usd", (0.0, -0.0115, 0.0)),  # colour camera
    "zed_x_mini": ("/Isaac/Sensors/Stereolabs/ZED_X_mini/ZED_X_Mini.usdc", (0.015, 0.025, 0.015)),  # left lens
}


def _camera(stage, path, eye_or_pos, quat: Gf.Quatf, focal: float, resolution) -> UsdGeom.Camera:
    cam = UsdGeom.Camera.Define(stage, path)
    h, w = resolution
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))  # also hides the camera's own housing
    set_pose(cam, eye_or_pos, quat)
    return cam


def _sensor_body(stage, assets_root: str, cam_path: str, model: str) -> None:
    """Put a real camera model around the USD camera at ``cam_path``, its lens on the camera origin.

    Only the meshes are used: the asset's own cameras, IMU, render products and physics are switched off."""
    asset, lens = SENSOR_MODELS[model]
    # Rows = asset axes in the camera frame: optical axis +x -> -z, +y -> -x, up +z -> +y.
    rot = Gf.Matrix3d(0, 0, -1, -1, 0, 0, 0, 1, 0)
    q = rot.ExtractRotation().GetQuat()
    body = UsdGeom.Xform.Define(stage, f"{cam_path}/body")
    set_pose(body, -(Gf.Vec3d(*lens) * rot), Gf.Quatf(q.GetReal(), *q.GetImaginary()))
    prim = UsdGeom.Xform.Define(stage, f"{cam_path}/body/model").GetPrim()  # keeps the asset's root transform
    prim.GetReferences().AddReference(assets_root + asset)
    for p in Usd.PrimRange(prim):
        p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        p.RemoveAPI(UsdPhysics.MassAPI)
        if p.IsA(UsdGeom.Camera) or p.GetTypeName() == "IsaacImuSensor" or p.GetName() == "TemplateRenderProducts":
            p.SetActive(False)


def _build_cameras(stage, assets_root: str, layout: Layout) -> None:
    ix, top = layout.inspect_x, layout.belt_top_z
    post_y = layout.frame_half_width + 0.15
    alu = color_material(stage, "aluminium", (0.75, 0.76, 0.78), 0.35, 0.8)

    # Inspection station: one camera hanging from the gantry arm, one on the post looking at the +y side.
    cam_z = top + 1.15
    eye = (ix, 0.0, cam_z)
    _camera(stage, "/World/Cameras/overhead", eye, look_at_quat(eye, (ix, 0.0, top), up=(0, 1, 0)),
            *CAMERAS["overhead"][::-1])
    _sensor_body(stage, assets_root, "/World/Cameras/overhead", "d455")
    cuboid(stage, "/World/Line/overhead_cam_stem", (ix, 0.0, cam_z + 0.05), (0.02, 0.02, 0.05), alu)
    eye = (ix, post_y - 0.1, top + 0.38)
    _camera(stage, "/World/Cameras/side", eye, look_at_quat(eye, (ix, 0.0, top + 0.1)), *CAMERAS["side"][::-1])
    _sensor_body(stage, assets_root, "/World/Cameras/side", "d455")
    cuboid(stage, "/World/Line/side_cam_bracket", (ix, post_y - 0.055, top + 0.395), (0.03, 0.06, 0.015), alu)

    # Wrist camera beside the suction tube, looking down the tool (+z of panda_hand) but tilted toward it so the
    # cup sits at the bottom edge of the image, like the gripper fingers in DROID's wrist view.
    side, tilt = np.array([math.cos(WRIST_CAM_AZIMUTH), math.sin(WRIST_CAM_AZIMUTH), 0.0]), math.radians(WRIST_CAM_TILT_DEG)
    eye = WRIST_CAM_OFFSET * side + (0.0, 0.0, 0.05)
    look = -math.sin(tilt) * side + (0.0, 0.0, math.cos(tilt))
    up = math.cos(tilt) * side + (0.0, 0.0, math.sin(tilt))  # image top points away from the tool
    _camera(stage, f"{HAND_PATH}/wrist_cam", tuple(eye), look_at_quat(eye, eye + look, up=up),
            *CAMERAS["wrist"][::-1])
    _sensor_body(stage, assets_root, f"{HAND_PATH}/wrist_cam", "zed_x_mini")
    dark = color_material(stage, "dark_metal", (0.08, 0.08, 0.09), 0.4, 0.6)
    cuboid(stage, f"{HAND_PATH}/wrist_cam_bracket", tuple(0.062 * side + (0.0, 0.0, 0.008)), (0.065, 0.1, 0.008),
           dark, yaw=WRIST_CAM_AZIMUTH)

    # Frames the line, the robot and the reject station from the front-left.
    cx, cy = layout.ng_table_center
    target = ((ix + cx) / 2 - 0.2, cy / 2, 0.4)
    eye = (target[0] - 3.0, target[1] - 3.6, 3.2)
    _camera(stage, "/World/Cameras/overview", eye, look_at_quat(eye, target), *CAMERAS["overview"][::-1])

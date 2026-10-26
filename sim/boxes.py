"""USD authoring of the procedural cardboard boxes: deformed face meshes (dents / tears), per-face
textures from :mod:`box_textures`, and a box-shaped rigid body.

Ground truth (``inspection:*`` attributes) is stored on the prim for evaluation only; the VLM never
sees it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from box_textures import FACE_FRAMES, FACES, BoxSpec, face_extent, face_point, render_box
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

GRID_SPACING = 0.008  # m between mesh vertices; dents need a fairly fine grid to shade smoothly


def _deform(points: np.ndarray, spec: BoxSpec) -> None:
    """Push vertices inward around dents and tears.

    The displacement depends only on the 3D vertex position, so vertices shared by two faces along an
    edge move identically and a crushed edge stays watertight.
    """
    rng = np.random.default_rng(spec.seed)
    for m in spec.defects:
        if m.depth <= 0:
            continue
        center = face_point(spec.size, m.face, m.uv)
        normal = np.asarray(FACE_FRAMES[m.face][0], float)
        radius = m.size / 2 if m.kind == "dent" else 0.35 * m.size
        offset = points - center
        dist = np.linalg.norm(offset, axis=-1)
        # Irregular rim: modulate the radius with a few random lobes around the dent centre.
        _, u_ax, v_ax = (np.asarray(a, float) for a in FACE_FRAMES[m.face])
        ang = np.arctan2(offset @ v_ax, offset @ u_ax)
        lobes = 1.0 + sum(rng.uniform(0.04, 0.12) * np.sin(k * ang + rng.uniform(0, 2 * np.pi)) for k in (2, 3, 5))
        w = 0.5 * (1 + np.cos(np.pi * np.clip(dist / (radius * lobes), 0, 1)))
        if m.kind == "dent":
            w = w ** 0.8 * (1 + 0.15 * np.sin(dist / radius * 9.0))  # a couple of soft wrinkles
        points -= normal * (m.depth * np.clip(w, 0, 1.2))[..., None]


def _face_mesh(stage, path: str, spec: BoxSpec, face: str) -> UsdGeom.Mesh:
    n, u, v = (np.asarray(a, float) for a in FACE_FRAMES[face])
    fu, fv = face_extent(spec.size, face)
    nu, nv = max(2, math.ceil(fu / GRID_SPACING)), max(2, math.ceil(fv / GRID_SPACING))
    s, t = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
    center = (np.abs(n) @ (np.asarray(spec.size) / 2)) * n
    pts = center + (s[..., None] - 0.5) * fu * u + (t[..., None] - 0.5) * fv * v
    _deform(pts, spec)
    normals = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12

    a = (np.arange(nv)[:, None] * (nu + 1) + np.arange(nu)[None, :]).ravel()
    indices = np.stack([a, a + 1, a + nu + 2, a + nu + 1], -1).ravel()

    mesh = UsdGeom.Mesh.Define(stage, path)
    flat = pts.reshape(-1, 3).astype(np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(flat))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, flat.min(0))), Gf.Vec3f(*map(float, flat.max(0)))]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (nu * nv)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices.astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(np.stack([s, t], -1).reshape(-1, 2).astype(np.float32)))
    return mesh


def textured_material(stage, path: str, texture: str | Path, roughness: float = 0.85) -> UsdShade.Material:
    """UsdPreviewSurface material with an sRGB diffuse texture read through the ``st`` primvar."""
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"{path}/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/diffuse")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def physics_material(stage, path: str, static: float, dynamic: float, restitution: float = 0.0):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(restitution)
    return mat


def spawn_box(
    stage,
    spec: BoxSpec,
    path: str,
    texture_dir: str | Path,
    position: tuple[float, float, float],
    yaw_deg: float = 0.0,
    mass: float = 0.8,
    physics_mat: UsdShade.Material | None = None,
) -> str:
    """Create the box rigid body at ``path``; ``position`` is the box centre in world coordinates."""
    textures = render_box(spec, texture_dir)
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    root.AddOrientOp().Set(Gf.Quatf(math.cos(math.radians(yaw_deg) / 2), 0, 0, math.sin(math.radians(yaw_deg) / 2)))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    collider = UsdGeom.Cube.Define(stage, f"{path}/collision")
    collider.CreateSizeAttr(1.0)
    collider.AddScaleOp().Set(Gf.Vec3f(*spec.size))
    collider.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    if physics_mat is not None:
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
            physics_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    UsdGeom.Xform.Define(stage, f"{path}/visual")
    for face in FACES:
        mesh = _face_mesh(stage, f"{path}/visual/{face}", spec, face)
        mat = textured_material(stage, f"{path}/looks/{face}", textures[face])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    prim.CreateAttribute("inspection:isDefective", Sdf.ValueTypeNames.Bool).Set(spec.is_defective)
    prim.CreateAttribute("inspection:spec", Sdf.ValueTypeNames.String).Set(json.dumps(spec.to_dict()))
    return path

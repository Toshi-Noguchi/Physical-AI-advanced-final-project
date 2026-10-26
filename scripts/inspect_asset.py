"""Print bounding box, prim tree and joints of Isaac assets (debug helper).

    docker/run.sh scripts/inspect_asset.py /Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("assets", nargs="+", help="paths relative to the Isaac assets root")
parser.add_argument("--depth", type=int, default=4)
args, _ = parser.parse_known_args()
app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

for asset in args.assets:
    stage_utils.create_new_stage()
    stage_utils.add_reference_to_stage(usd_path=get_assets_root_path() + asset, path="/World/asset")
    while stage_utils.is_stage_loading():
        app.update()
    stage = stage_utils.get_current_stage()
    root = stage.GetPrimAtPath("/World/asset")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root)
    rng = bbox.ComputeAlignedRange()
    print("ASSET", asset, "BBOX", rng.GetMin(), rng.GetMax())
    for prim in Usd.PrimRange(root):
        depth = prim.GetPath().pathElementCount - 2
        apis = [a for a in prim.GetAppliedSchemas() if "Physics" in a or "Isaac" in a]
        extra = ""
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            extra = f" body0={j.GetBody0Rel().GetTargets()} body1={j.GetBody1Rel().GetTargets()}"
        if prim.IsA(UsdGeom.Camera):  # where the lens sits and where it looks, in the asset's frame
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            extra = (f" pos={m.ExtractTranslation()} look={m.TransformDir((0, 0, -1)).GetNormalized()}"
                     f" up={m.TransformDir((0, 1, 0)).GetNormalized()}")
        if depth <= args.depth or prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdGeom.Camera):
            print("  " * depth + f"{prim.GetName()} [{prim.GetTypeName()}] {apis}{extra}")
app.close()

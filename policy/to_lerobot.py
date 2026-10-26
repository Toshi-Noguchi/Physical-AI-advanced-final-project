"""Convert raw demonstrations (scripts/collect_demos.py) into a LeRobot v3.0 dataset for SmolVLA fine-tuning.

    docker/convert.sh data/raw/run1 --out data/lerobot/pi_final_suction_pick

Keys follow lerobot/smolvla_base, so it fine-tunes without renaming: observation.images.camera1 (overhead),
camera2 (side), camera3 (wrist) as videos; observation.state and action as float32[8] (7 Franka joint positions
+ gripper; action = absolute joint targets + suction command). Per-episode extras (raw episode name, box spec,
defect flag, plan parameters) go to meta/pi_final_episodes.jsonl next to LeRobot's own metadata.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

CAMERAS = {"overhead": "camera1", "side": "camera2", "wrist": "camera3"}  # raw name -> dataset key
JOINT_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


def features_for(episode: Path) -> tuple[int, dict]:
    meta = json.loads((episode / "meta.json").read_text())
    features = {
        f"observation.images.{key}": {"dtype": "video", "shape": (*meta["cameras"][cam], 3),
                                      "names": ["height", "width", "channels"]}
        for cam, key in CAMERAS.items()
    }
    features["observation.state"] = {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES}
    features["action"] = {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES}
    return int(meta["fps"]), features


def convert(episodes: list[Path], out: Path, repo_id: str, threads: int, tag: str = "") -> list[dict]:
    """Write ``episodes`` as one LeRobot dataset at ``out``; returns the per-episode extras."""
    fps, features = features_for(episodes[0])
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=out,
                               robot_type="franka_panda_on_ridgeback_suction", use_videos=True,
                               image_writer_threads=threads)
    extras = []
    for n, ep in enumerate(episodes):
        meta = json.loads((ep / "meta.json").read_text())
        data = np.load(ep / "episode.npz")
        assert meta["fps"] == fps and len(data["state"]) == meta["frames"], ep
        for i in range(meta["frames"]):
            frame = {f"observation.images.{key}": np.asarray(Image.open(ep / cam / f"{i:06d}.jpg").convert("RGB"))
                     for cam, key in CAMERAS.items()}
            frame["observation.state"] = data["state"][i].astype(np.float32)
            frame["action"] = data["action"][i].astype(np.float32)
            frame["task"] = meta["task"]
            ds.add_frame(frame)
        ds.save_episode()
        extras.append({"episode_index": n, "raw_episode": ep.name, "task": meta["task"],
                       "box_defective": meta["box_defective"], "box": meta["box"], "plan": meta["plan"],
                       "lift_height": meta["lift_height"], "seed": meta["seed"]})
        print(f"[convert]{tag} {n + 1}/{len(episodes)} {ep.name} ({meta['frames']} frames)", flush=True)
    ds.finalize()
    return extras


def _convert_shard(episodes, out, repo_id, threads, tag, extras_path) -> None:
    extras_path.write_text(json.dumps(convert(episodes, out, repo_id, threads, tag)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", nargs="+", help="raw collection directories")
    parser.add_argument("--out", default="data/lerobot/pi_final_suction_pick")
    parser.add_argument("--repo-id", default="local/pi_final_suction_pick")
    parser.add_argument("--limit", type=int, default=0, help="convert only the first N episodes (0 = all)")
    parser.add_argument("--workers", type=int, default=1, help="convert shards in parallel, then merge them")
    parser.add_argument("--threads", type=int, default=8, help="image writer threads per worker")
    args = parser.parse_args()

    # An episode is complete once its meta.json exists (it is written last).
    episodes = sorted(p.parent for raw in args.raw for p in Path(raw).glob("w*_c*_e*/meta.json"))
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit("no complete episodes found")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    if args.workers <= 1:
        extras = convert(episodes, out, args.repo_id, args.threads)
    else:
        # Contiguous shards keep the episode order; aggregate_datasets renumbers them in shard order.
        bounds = np.linspace(0, len(episodes), args.workers + 1).astype(int)
        shard_dir = out.parent / f"_{out.name}_shards"
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        jobs = [(episodes[a:b], shard_dir / f"shard{k}", f"{args.repo_id}_shard{k}", args.threads, f" [shard {k}]",
                 shard_dir / f"shard{k}_extras.json") for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
        shard_dir.mkdir(parents=True)
        # Plain (non-daemon) processes: LeRobot's writer starts its own encoder processes.
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_convert_shard, args=job) for job in jobs]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        if any(proc.exitcode for proc in procs):
            raise SystemExit(f"shard conversion failed: exit codes {[proc.exitcode for proc in procs]}")
        shard_extras = [json.loads(job[5].read_text()) for job in jobs]
        aggregate_datasets(repo_ids=[job[2] for job in jobs], aggr_repo_id=args.repo_id,
                           roots=[job[1] for job in jobs], aggr_root=out)
        extras, offset = [], 0
        for part in shard_extras:
            extras += [{**e, "episode_index": e["episode_index"] + offset} for e in part]
            offset += len(part)
        shutil.rmtree(shard_dir)

    with open(out / "meta" / "pi_final_episodes.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in extras)
    (out / "meta" / "pi_final_info.json").write_text(json.dumps(
        {"cameras": {key: cam for cam, key in CAMERAS.items()}, "state_action_names": JOINT_NAMES,
         "action": "absolute joint position targets (rad) + suction command (0 off / 1 on)",
         "gripper_state": "0 off, 0.5 holding a box, 0.8 suction on but empty"}, indent=1))

    # Read it back as the trainer would.
    check = LeRobotDataset(args.repo_id, root=out)
    item = check[len(check) // 2]
    shapes = {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")}
    print(f"[convert] done: {check.num_episodes} episodes, {check.num_frames} frames at {check.fps} fps -> {out}")
    print(f"[convert] sample frame: {shapes} task={item['task']!r}")
    last = check[len(check) - 1]
    assert int(last["episode_index"]) == len(extras) - 1, "episode count mismatch after merging"


if __name__ == "__main__":
    main()

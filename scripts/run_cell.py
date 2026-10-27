"""The inspection cell end to end. Boxes flow down the line and stop at the station; Molmo2 inspects them. Good boxes
go on; for a defective one the arm moves home, the grasp VLA (SmolVLA) picks it, Molmo2 checks the pick from the
station's side camera and the wrist camera (retrying if it failed), and rule-based skills carry the box to the reject
table, put it down and bring the robot back to the line.

    scripts/run_cell.sh --boxes 12 --record --out outputs/cell     # starts both model servers, then this script

or by hand, each server in a terminal of its own (the controller waits for servers that are still loading):

    docker/vlm.sh policy/vlm_server.py                    # Molmo2, 127.0.0.1:6124
    docker/policy.sh outputs/models/smolvla_pick          # SmolVLA, 127.0.0.1:6123
    docker/run.sh scripts/run_cell.py --boxes 12 --record --out outputs/cell

Without the model servers (development): --inspector oracle (ground truth), --grasp scripted (the expert that made
the demonstrations), --check sensor (the suction gripper's own state).
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--boxes", type=int, default=10, help="boxes to inspect before stopping")
parser.add_argument("--p-defect", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--inspector", choices=("molmo2", "oracle"), default="molmo2")
parser.add_argument("--grasp", choices=("vla", "scripted"), default="vla")
parser.add_argument("--check", choices=("vlm", "sensor"), default="vlm", help="who decides whether the pick worked")
parser.add_argument("--task", default="pick up the cardboard box", help="instruction given to the grasp VLA")
parser.add_argument("--attempts", type=int, default=2, help="grasp attempts before giving up on a box")
parser.add_argument("--fallback-scripted", action="store_true", help="after failed VLA attempts, try the expert")
parser.add_argument("--record", action="store_true", help="save overview frames with a status overlay (30 fps)")
parser.add_argument("--server-wait", type=float, default=600.0, help="seconds to wait for the model servers")
parser.add_argument("--out", default="outputs/cell")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "policy")]

import numpy as np  # noqa: E402
from env import InspectionLineEnv  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from protocol import POLICY_ADDRESS, VLM_ADDRESS, ModelClient  # noqa: E402
from skills import (RejectTable, arm_home, carry_to_table, place, retract_for_inspection, return_to_line,  # noqa: E402
                    scripted_pick, vla_pick)

GRASP_QUESTION = ("Image 1 shows a conveyor inspection station from the side. Image 2 is the robot's wrist camera, "
                  "looking down past its blue suction cup. Is a cardboard box lifted up off the conveyor and held "
                  "by the suction cup?")


class Recorder:
    """Overview frames at a fixed sim-time rate, with the controller's state written over them."""

    def __init__(self, out: Path, fps: int = 30):
        self.dir, self.period, self.next_t, self.n = out / "frames", 1.0 / fps, 0.0, 0
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []
        try:
            self.font = ImageFont.truetype(str(ROOT / "assets/fonts/DejaVuSans-Bold.ttf"), 22)
        except OSError:
            self.font = ImageFont.load_default()

    def __call__(self, env: InspectionLineEnv) -> None:
        if env.time < self.next_t:
            return
        frame = env.capture("overview")
        if frame is None:  # the renderer has not delivered this camera's first frame yet: try again next step
            return
        self.next_t = env.time + self.period
        img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((16, 16, 16 + 560, 24 + 30 * len(self.lines)), fill=(10, 14, 20, 170))
        for i, line in enumerate(self.lines):
            draw.text((28, 22 + 30 * i), line, font=self.font, fill=(255, 255, 255, 255))
        Image.alpha_composite(img, overlay).convert("RGB").save(self.dir / f"{self.n:06d}.jpg", quality=88)
        self.n += 1


class Cell:
    def __init__(self):
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = InspectionLineEnv(app, self.out / "_env", seed=args.seed, p_defect=args.p_defect,
                                     cameras=("overhead", "side", "wrist", "overview"))
        self.table = RejectTable(self.env.layout)
        self.rng = np.random.default_rng(args.seed)
        need_vlm = args.inspector == "molmo2" or args.check == "vlm"
        self.vlm = ModelClient(VLM_ADDRESS, wait=args.server_wait) if need_vlm else None
        self.vla = ModelClient(POLICY_ADDRESS, wait=args.server_wait) if args.grasp == "vla" else None
        self.rec = Recorder(self.out) if args.record else None
        self.counts = {"inspected": 0, "passed": 0, "rejected": 0, "placed": 0}
        self.state, self.box_line, self.verdict_line = "", "", ""
        self.log = open(self.out / "boxes.jsonl", "w")

    # ---------------------------------------------------------------- bookkeeping
    def set_state(self, state: str) -> None:
        self.state = state
        print(f"[cell] t={self.env.time:6.1f}s {state}", flush=True)
        if self.rec:
            c = self.counts
            self.rec.lines = [f"State: {state}", self.box_line, self.verdict_line,
                              f"Inspected {c['inspected']}  Passed {c['passed']}  Rejected {c['rejected']}"]

    def on_step(self, env) -> None:
        if self.rec:
            self.rec(env)

    def step(self, n: int = 1, auto_spawn: bool = True) -> None:
        for _ in range(n):
            self.env.step(1, auto_spawn=auto_spawn)
            self.on_step(self.env)

    # ---------------------------------------------------------------- decisions
    def inspect(self, path: str) -> dict:
        top, side = self.env.capture_inspection()
        spec = self.env.boxes[path]["spec"]
        if args.inspector == "oracle":
            return {"defective": spec.is_defective, "verdict": "DEFECTIVE" if spec.is_defective else "OK",
                    "reason": ", ".join(m.kind for m in spec.defects), "latency": 0.0}
        return self.vlm.inspect([top, side])

    def picked(self, path: str, start_z: float) -> dict:
        """Did the pick work? The sensor answer (suction holding this box, lifted) is always logged."""
        env = self.env
        sensor = env.held_box() == path and env.box_pose(path)[0][2] > start_z + 0.05
        if args.check == "sensor":
            return {"picked": sensor, "sensor": sensor}
        reply = self.vlm.call("ask", images=[env.capture("side"), env.capture("wrist")], question=GRASP_QUESTION)
        return {"picked": bool(reply["answer"]), "sensor": sensor, "vlm_raw": reply["raw"], "latency": reply["latency"]}

    # ---------------------------------------------------------------- the line
    def run(self) -> None:
        env = self.env
        env.set_belt(False)
        self.step(30, auto_spawn=False)
        self.set_state("Pulling the arm back for inspection")
        retract_for_inspection(env, on_step=self.on_step)
        env.spawn_box()
        env.set_belt(True)
        self.set_state("Line running")
        while self.counts["inspected"] < args.boxes:
            self.step()
            if env.waiting_box:
                self.handle(env.waiting_box)
                self.set_state("Line running")
        self.step(60)
        summary = {**self.counts, "sim_time": round(env.time, 1), "args": vars(args)}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[cell] done: {summary}", flush=True)

    def handle(self, path: str) -> None:
        env = self.env
        spec = env.boxes[path]["spec"]
        self.step(15, auto_spawn=False)  # let the box settle under the cameras
        self.box_line, self.verdict_line = f"Box: {spec.box_id}", ""
        self.set_state("Inspecting (Molmo2)" if args.inspector == "molmo2" else "Inspecting (ground truth)")
        t0 = time.time()
        verdict = self.inspect(path)
        self.counts["inspected"] += 1
        self.verdict_line = f"Verdict: {verdict['verdict']}" + (f" ({verdict['reason']})" if verdict["reason"] else "")
        entry = {"box": spec.box_id, "truth_defective": spec.is_defective,
                 "truth_defects": [(m.kind, m.variant, m.face) for m in spec.defects],
                 "verdict": verdict["verdict"], "reason": verdict["reason"], "inspect_s": round(time.time() - t0, 2),
                 "sim_t": round(env.time, 1)}
        self.set_state(f"Verdict: {verdict['verdict']}")
        self.step(20, auto_spawn=False)
        if not verdict["defective"]:
            self.counts["passed"] += 1
            self.set_state("OK: passing the box on")
            env.release_box()
            self.write(entry)
            return

        self.counts["rejected"] += 1
        start_z = env.box_pose(path)[0][2]
        attempts = []
        picked = False
        for attempt in range(1, args.attempts + 1):
            self.set_state("Moving to the home pose")
            arm_home(env, on_step=self.on_step)
            method = args.grasp
            if method == "vla":
                self.set_state(f"Grasping with SmolVLA (attempt {attempt})")
                result = vla_pick(env, self.vla, args.task, on_step=self.on_step)
            else:
                self.set_state(f"Grasping (scripted, attempt {attempt})")
                result = {"finished": scripted_pick(env, path, self.rng, on_step=self.on_step)}
            self.set_state("Checking the pick (Molmo2)" if args.check == "vlm" else "Checking the pick (sensor)")
            check = self.picked(path, start_z)
            attempts.append({"method": method, **result, **check})
            if check["picked"]:
                picked = True
                break
            self.set_state("Pick failed: letting go and retrying")
            env.release()
        if not picked and args.fallback_scripted and args.grasp == "vla":
            self.set_state("Falling back to the scripted grasp")
            arm_home(env, on_step=self.on_step)
            scripted_pick(env, path, self.rng, on_step=self.on_step)
            check = self.picked(path, start_z)
            attempts.append({"method": "scripted", **check})
            picked = check["picked"]
        entry["attempts"] = attempts

        if picked:
            env.take_off_line(path)
            self.set_state("Carrying to the reject table")
            slot = carry_to_table(env, self.table, on_step=self.on_step)
            self.set_state("Putting the box down")
            place(env, self.table, slot, path, on_step=self.on_step)
            self.counts["placed"] += 1
            self.set_state("Returning to the line")
            return_to_line(env, self.table, slot, on_step=self.on_step)
            box_pos = env.box_pose(path)[0]
            entry.update(slot=slot, placed_at=box_pos.round(3).tolist())
        else:
            self.set_state("Could not pick the box: letting it pass")
            env.release()
            arm_home(env, on_step=self.on_step)
            env.release_box()
        self.set_state("Pulling the arm back for inspection")
        retract_for_inspection(env, on_step=self.on_step)
        env.set_belt(True)
        self.write(entry)

    def write(self, entry: dict) -> None:
        self.log.write(json.dumps(entry) + "\n")
        self.log.flush()
        print(f"[cell] {entry['box']}: truth={'NG' if entry['truth_defective'] else 'OK'} verdict={entry['verdict']} "
              f"attempts={[(a['method'], a['picked'], a['sensor']) for a in entry.get('attempts', [])]}", flush=True)

    def close(self) -> None:
        self.log.close()
        for client in (self.vlm, self.vla):
            if client:
                client.close()


cell = None
try:
    cell = Cell()  # inside the try: a server that never comes up must still end in app.close()
    cell.run()
except Exception:
    traceback.print_exc()  # app.close() ends the process before Python would print it
    raise
finally:
    if cell:
        cell.close()
    app.close()

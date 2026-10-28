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
import shutil
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
parser.add_argument("--attempts", type=int, default=3, help="grasp attempts before giving up on a box")
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

# The video is captioned in Japanese; docker/run.sh mounts the host's Noto CJK fonts when it has them.
FONT_JA = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
DEFECTS_JA = {"ng": "NG 表示", "reject": "REJECT 表示", "tear": "破れ", "dent": "凹み"}
LEGEND_JA = "上位: Molmo2（VLM）が検査・状態遷移・把持の確認　／　下位: SmolVLA（VLA）が把持　／　搬送・置く・復帰: ルールベース"


def fit(img: np.ndarray, w: int, h: int) -> Image.Image:
    """Center-crop an image to the w:h aspect and scale it to w x h."""
    ih, iw = img.shape[:2]
    if iw * h > ih * w:
        cw = ih * w // h
        img = img[:, (iw - cw) // 2:(iw + cw) // 2]
    else:
        ch = iw * h // w
        img = img[(ih - ch) // 2:(ih + ch) // 2]
    return Image.fromarray(np.ascontiguousarray(img)).resize((w, h), Image.Resampling.BILINEAR)


class Recorder:
    """1920x1080 video frames at a fixed sim-time rate: the overview camera, the grasp policy's wrist and side cameras,
    the top view Molmo2 last judged, and the controller's state in Japanese."""

    def __init__(self, out: Path, fps: int = 30):
        self.dir, self.fps, self.period, self.next_t, self.n = out / "frames", fps, 1.0 / fps, 0.0, 0
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state, self.box, self.verdict, self.check, self.counts = "", "", "", "", ""
        self.inspection: tuple[np.ndarray, bool | None] | None = None  # (top view as Molmo2 saw it, defective)
        self.views: dict[str, np.ndarray] | None = None
        try:
            self.font = {size: ImageFont.truetype(FONT_JA, size) for size in (20, 26, 30, 44)}
        except OSError:
            self.font = {size: ImageFont.truetype(str(ROOT / "assets/fonts/DejaVuSans-Bold.ttf"), size)
                         for size in (20, 26, 30, 44)}

    def __call__(self, env: InspectionLineEnv) -> None:
        if env.time < self.next_t:
            return
        views = {name: env.capture(name) for name in ("overview", "wrist", "side")}
        if any(v is None for v in views.values()):  # the renderer has not delivered a first frame yet: next step
            return
        self.next_t = env.time + self.period
        self.views = views
        self.compose().save(self.dir / f"{self.n:06d}.jpg", quality=88)
        self.n += 1

    def hold(self, seconds: float) -> None:
        """Repeat the current picture (with the current captions) while the line waits for a model."""
        if self.views is None:
            return
        first = self.dir / f"{self.n:06d}.jpg"
        self.compose().save(first, quality=88)
        self.n += 1
        for _ in range(round(seconds * self.fps) - 1):
            shutil.copyfile(first, self.dir / f"{self.n:06d}.jpg")
            self.n += 1

    def label(self, draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(10, 14, 20, 190)) -> None:
        x, y = xy
        box = draw.textbbox((x + 12, y + 8), text, font=self.font[20])
        draw.rectangle((x, y, box[2] + 12, box[3] + 8), fill=fill)
        draw.text((x + 12, y + 8), text, font=self.font[20], fill=(255, 255, 255))

    def compose(self) -> Image.Image:
        v = self.views
        frame = Image.new("RGB", (1920, 1080), (16, 20, 27))
        frame.paste(Image.fromarray(v["overview"]).resize((1280, 720), Image.Resampling.BILINEAR), (0, 0))
        frame.paste(fit(v["wrist"], 640, 360), (1280, 0))
        frame.paste(fit(v["side"], 640, 360), (1280, 360))
        if self.inspection is not None:
            frame.paste(fit(self.inspection[0], 640, 360), (1280, 720))
        draw = ImageDraw.Draw(frame, "RGBA")
        self.label(draw, (0, 0), "全体の様子")
        self.label(draw, (1280, 0), "手首カメラ（SmolVLA の入力）")
        self.label(draw, (1280, 360), "側面カメラ（SmolVLA の入力）")
        self.label(draw, (1280, 720), "検査画像・真上（Molmo2 の入力）")
        if self.inspection is None:
            draw.text((1600, 900), "検査待ち", font=self.font[30], fill=(150, 160, 175), anchor="mm")
        elif self.inspection[1] is not None:
            ok = not self.inspection[1]
            self.label(draw, (1280, 1080 - 46), f"判定: {'良品' if ok else '不良'}",
                       fill=(30, 140, 70, 220) if ok else (200, 40, 40, 220))
        draw.line((1280, 0, 1280, 1080), fill=(16, 20, 27), width=4)
        for y in (360, 720):
            draw.line((1280, y, 1920, y), fill=(16, 20, 27), width=4)
        draw.text((40, 742), "状態", font=self.font[20], fill=(150, 160, 175))
        draw.text((40, 768), self.state, font=self.font[44], fill=(255, 255, 255))
        draw.text((40, 842), "　".join(t for t in (self.box, self.verdict) if t), font=self.font[30],
                  fill=(225, 230, 238))
        draw.text((40, 888), self.check, font=self.font[30], fill=(225, 230, 238))
        draw.text((40, 954), self.counts, font=self.font[26], fill=(255, 210, 90))
        draw.text((40, 1030), LEGEND_JA, font=self.font[20], fill=(150, 160, 175))
        return frame


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
        self.state = ""
        self.log = open(self.out / "boxes.jsonl", "w")

    # ---------------------------------------------------------------- bookkeeping
    def set_state(self, state: str, caption: str) -> None:
        """``state`` goes to the log, ``caption`` (Japanese) onto the video."""
        self.state = state
        frame = f" frame={self.rec.n}" if self.rec else ""  # where the video is, for editing
        print(f"[cell] t={self.env.time:6.1f}s{frame} {state}", flush=True)
        if self.rec:
            c = self.counts
            self.rec.state = caption
            self.rec.counts = (f"検査 {c['inspected']}　｜　良品 {c['passed']}　｜　不良 {c['rejected']}"
                               f"　｜　置き場へ {c['placed']}")

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
        shots = self.out / "inspection"
        shots.mkdir(exist_ok=True)
        box_id = self.env.boxes[path]["spec"].box_id
        Image.fromarray(top).save(shots / f"{box_id}_top.png")
        Image.fromarray(side).save(shots / f"{box_id}_side.png")
        if self.rec:
            h, w = top.shape[:2]  # the central half Molmo2 is shown (policy/vlm_server.py --top-crop 0.5)
            self.rec.inspection = (top[h // 4:h - h // 4, w // 4:w - w // 4], None)
        spec = self.env.boxes[path]["spec"]
        if args.inspector == "oracle":
            return {"defective": spec.is_defective, "verdict": "DEFECTIVE" if spec.is_defective else "OK",
                    "reason": ", ".join(m.kind for m in spec.defects), "latency": 0.0}
        return self.vlm.inspect([top, side])

    def picked(self, path: str, start_z: float) -> dict:
        """Did the pick work? Molmo2 must see the box lifted *and* the cup's vacuum switch must report a seal: with
        the cup hovering just above the box top, Molmo2 alone once said yes to an empty cup. The privileged answer
        (this box held and lifted, from the simulator) is only logged."""
        env = self.env
        sensor = bool(env.held_box() == path and env.box_pose(path)[0][2] > start_z + 0.05)
        vacuum = env.held_box() is not None  # what a real suction cup's vacuum switch tells
        if args.check == "sensor":
            return {"picked": sensor, "sensor": sensor, "latency": 0.0}
        reply = self.vlm.call("ask", images=[env.capture("side"), env.capture("wrist")], question=GRASP_QUESTION)
        return {"picked": bool(reply["answer"]) and vacuum, "vlm": reply["answer"], "vacuum": vacuum, "sensor": sensor,
                "vlm_raw": reply["raw"], "latency": reply["latency"]}

    def wait_for_model(self, latency: float) -> None:
        """On the video, hold the picture for as long as the model took (at least a second, to be readable)."""
        if self.rec:
            self.rec.hold(max(latency, 1.0))

    # ---------------------------------------------------------------- the line
    def run(self) -> None:
        env = self.env
        env.set_belt(False)
        self.step(30, auto_spawn=False)
        self.set_state("Pulling the arm back for inspection", "検査の邪魔にならないよう腕を退避")
        retract_for_inspection(env, on_step=self.on_step)
        env.spawn_box()
        env.set_belt(True)
        self.set_state("Line running", "ライン稼働中: 箱を検査位置へ")
        while self.counts["inspected"] < args.boxes:
            self.step()
            if env.waiting_box:
                self.handle(env.waiting_box)
                self.set_state("Line running", "ライン稼働中: 箱を検査位置へ")
        self.step(60)
        summary = {**self.counts, "sim_time": round(env.time, 1), "args": vars(args)}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[cell] done: {summary}", flush=True)

    def handle(self, path: str) -> None:
        env, rec = self.env, self.rec
        spec = env.boxes[path]["spec"]
        self.step(15, auto_spawn=False)  # let the box settle under the cameras
        if rec:
            rec.box, rec.verdict, rec.check, rec.inspection = f"箱 {spec.box_id}", "", "", None
        if args.inspector == "molmo2":
            self.set_state("Inspecting (Molmo2)", "Molmo2 が外観を検査（斜め照明で撮影）")
        else:
            self.set_state("Inspecting (ground truth)", "外観検査（正解ラベルを使用）")
        t0 = time.time()
        verdict = self.inspect(path)
        self.counts["inspected"] += 1
        entry = {"box": spec.box_id, "truth_defective": spec.is_defective,
                 "truth_defects": [(m.kind, m.variant, m.face) for m in spec.defects],
                 "verdict": verdict["verdict"], "reason": verdict["reason"], "inspect_s": round(time.time() - t0, 2),
                 "sim_t": round(env.time, 1)}
        found = "・".join(DEFECTS_JA.get(k.strip(), k.strip()) for k in verdict["reason"].split(",") if k.strip())
        if rec:
            rec.inspection = (rec.inspection[0], verdict["defective"])
            rec.verdict = "Molmo2 の判定: " + (f"不良（{found}）" if verdict["defective"] else "良品")
        self.set_state(f"Verdict: {verdict['verdict']}",
                       f"判定: 不良（{found}）" if verdict["defective"] else "判定: 良品")
        self.wait_for_model(verdict["latency"])
        self.step(20, auto_spawn=False)
        if not verdict["defective"]:
            self.counts["passed"] += 1
            self.set_state("OK: passing the box on", "良品: そのまま流す")
            env.release_box()
            self.write(entry)
            return

        self.counts["rejected"] += 1
        start_z = env.box_pose(path)[0][2]
        attempts = []
        picked = False
        checker = "Molmo2 と吸着センサー" if args.check == "vlm" else "吸着センサー"
        for attempt in range(1, args.attempts + 1):
            self.set_state("Moving to the home pose", "腕をホーム姿勢へ（把持の開始位置）")
            arm_home(env, on_step=self.on_step)
            method = args.grasp
            if method == "vla":
                self.set_state(f"Grasping with SmolVLA (attempt {attempt})",
                               f"SmolVLA が吸着して持ち上げる（{attempt} 回目）")
                result = vla_pick(env, self.vla, args.task, on_step=self.on_step)
            else:
                self.set_state(f"Grasping (scripted, attempt {attempt})", f"お手本の動作で把持（{attempt} 回目）")
                result = {"finished": scripted_pick(env, path, self.rng, on_step=self.on_step)}
            self.set_state("Checking the pick (Molmo2)" if args.check == "vlm" else "Checking the pick (sensor)",
                           f"{checker}が把持できたかを確認")
            check = self.picked(path, start_z)
            attempts.append({"method": method, **result, **check})
            if rec:
                verdict_ja = "成功" if check["picked"] else "失敗"
                rec.check = (f"把持の確認: Molmo2 {'○' if check['vlm'] else '×'}　吸着センサー "
                             f"{'○' if check['vacuum'] else '×'}　→ {verdict_ja}" if "vlm" in check
                             else f"把持の確認（吸着センサー）: {verdict_ja}")
            if check["picked"]:
                self.set_state("Pick confirmed", "把持を確認")
                self.wait_for_model(check["latency"])
                picked = True
                break
            last = attempt == args.attempts
            self.set_state("Pick failed: letting go" if last else "Pick failed: letting go and retrying",
                           "把持に失敗: 離す" if last else "把持に失敗: 離して SmolVLA をもう一度起動")
            self.wait_for_model(check["latency"])
            env.release()
        if not picked and args.fallback_scripted and args.grasp == "vla":
            self.set_state("Falling back to the scripted grasp", "予備: お手本の動作で把持")
            arm_home(env, on_step=self.on_step)
            scripted_pick(env, path, self.rng, on_step=self.on_step)
            check = self.picked(path, start_z)
            attempts.append({"method": "scripted", **check})
            picked = check["picked"]
        entry["attempts"] = attempts

        if picked:
            env.take_off_line(path)
            self.set_state("Carrying to the reject table", "不良品置き場へ搬送")
            slot = carry_to_table(env, self.table, on_step=self.on_step)
            self.set_state("Putting the box down", "置き場に置く")
            place(env, self.table, slot, path, on_step=self.on_step)
            self.counts["placed"] += 1
            self.set_state("Returning to the line", "ラインへ戻る")
            return_to_line(env, self.table, slot, on_step=self.on_step)
            box_pos = env.box_pose(path)[0]
            entry.update(slot=slot, placed_at=box_pos.round(3).tolist())
        else:
            self.set_state("Could not pick the box: letting it pass", "把持できず: 箱を流す")
            env.release()
            arm_home(env, on_step=self.on_step)
            env.release_box()
        self.set_state("Pulling the arm back for inspection", "次の検査に備えて腕を退避")
        retract_for_inspection(env, on_step=self.on_step)
        env.set_belt(True)
        self.write(entry)

    def write(self, entry: dict) -> None:
        self.log.write(json.dumps(entry, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
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

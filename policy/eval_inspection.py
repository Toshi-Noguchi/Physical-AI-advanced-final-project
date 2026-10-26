"""Score the inspection VLM on a captured set (scripts/make_inspection_set.py).

    docker/vlm.sh policy/eval_inspection.py data/inspection/eval1 --model allenai/Molmo2-8B

Writes <set>/pred_<model>.jsonl and prints accuracy, recall per defect kind and which harmless marks caused
false alarms.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from PIL import Image
from vlm_inspect import Molmo2Inspector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("set_dir")
    parser.add_argument("--model", default="allenai/Molmo2-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--style", nargs="+", default=["checklist"], help="prompt styles to score (vlm_inspect.PROMPTS)")
    parser.add_argument("--top-crop", type=float, nargs="+", default=[0.5], help="central fraction of the top view (0 = full)")
    parser.add_argument("--exemplars", default=None, help="directory with index.json for the checklist_fewshot style")
    args = parser.parse_args()

    root = Path(args.set_dir)
    labels = [json.loads(line) for line in open(root / "labels.jsonl")]
    if args.limit:
        labels = labels[: args.limit]
    t0 = time.time()
    inspector = Molmo2Inspector(args.model, exemplars=args.exemplars)
    print(f"[eval] {args.model} loaded in {time.time() - t0:.0f}s", flush=True)

    for style, crop in ((s, c) for s in args.style for c in args.top_crop):
        preds = []
        tag = f"{style}_crop{crop:g}"
        out_path = root / f"pred_{args.model.split('/')[-1]}_{tag}.jsonl"
        with open(out_path, "w") as f:
            for lab in labels:
                i = lab["index"]
                images = [Image.open(root / f"{i:04d}_{cam}.png").convert("RGB") for cam in ("overhead", "side")]
                t = time.time()
                pred = inspector.judge(images, style, crop or None)
                pred.update(index=i, seconds=round(time.time() - t, 2))
                preds.append(pred)
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                f.flush()
                if i % 20 == 0:
                    print(f"[eval] {i}: truth={lab['defective']} pred={pred['verdict']} ({pred['seconds']}s) {pred['raw']!r}",
                          flush=True)

        # Scores: defective = positive.
        tp = sum(p["defective"] and lab["defective"] for p, lab in zip(preds, labels))
        fp = sum(p["defective"] and not lab["defective"] for p, lab in zip(preds, labels))
        fn = sum(not p["defective"] and lab["defective"] for p, lab in zip(preds, labels))
        tn = len(labels) - tp - fp - fn
        print(f"[eval] {tag}: n={len(labels)} acc={(tp + tn) / len(labels):.3f} recall={tp / max(1, tp + fn):.3f} "
              f"precision={tp / max(1, tp + fp):.3f} false-alarm rate={fp / max(1, fp + tn):.3f} "
              f"(tp={tp} fp={fp} fn={fn} tn={tn}) unknown={sum(p['verdict'] == 'UNKNOWN' for p in preds)} "
              f"mean {sum(p['seconds'] for p in preds) / len(preds):.2f}s/box", flush=True)
        seen, hit = Counter(), Counter()
        for p, lab in zip(preds, labels):
            for m in lab["defects"]:
                key = f"{m['kind']}/{m['variant']}@{m['face']}"
                seen[key] += 1
                hit[key] += p["defective"]
        for key in sorted(seen):
            print(f"[eval]   recall {key:32s} {hit[key]}/{seen[key]}", flush=True)
        alarms = Counter(m["kind"] for p, lab in zip(preds, labels) if p["defective"] and not lab["defective"]
                         for m in lab["distractors"])
        print(f"[eval] marks on falsely rejected boxes: {dict(alarms)}", flush=True)


if __name__ == "__main__":
    main()

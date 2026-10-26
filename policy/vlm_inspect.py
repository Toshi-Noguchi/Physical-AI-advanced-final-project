"""Visual inspection with Molmo2: is the box at the station defective?

The model sees the station's overhead and side images and answers DEFECTIVE / OK plus a short reason. Used by
the evaluation (policy/eval_inspection.py) and by the line's state machine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = """You are the visual inspector on a cardboard box conveyor line.
Image 1 is the top view of the box and image 2 is the side view.
The box is DEFECTIVE if it shows any of these:
- a red "NG" label or stamp, or a Japanese "不良" stamp
- a "REJECT" stamp or sticker
- a tear or hole in the cardboard
- a dent or a crushed edge or corner
These are normal and do NOT make a box defective: shipping labels with barcodes, "FRAGILE" stickers,
green "QC PASSED" stickers, "THIS SIDE UP" arrows, packing tape, printed part numbers.
Answer with one word on the first line, DEFECTIVE or OK. On the second line, name the mark or damage you saw."""

# Describe first, decide last: answering the verdict first made the model say OK and then name the defect.
PROMPT_DESCRIBE = """You are the visual inspector on a cardboard box conveyor line.
Image 1 is the top view of the box and image 2 is the side view.
First, in one sentence, list every label, stamp, sticker and any damage you can see on the box.
Then, on the last line, write DEFECTIVE or OK.
The box is DEFECTIVE if it has any of these: an "NG" label, stamp or handwritten mark; a Japanese "不良" stamp;
a "REJECT" stamp or sticker; a tear or hole in the cardboard; a dent or a crushed edge or corner.
These are normal and do NOT make a box defective: shipping labels with barcodes, "FRAGILE" stickers,
green "QC PASSED" stickers, "THIS SIDE UP" arrows, packing tape, printed part numbers."""

# One yes/no question per defect type; the code takes the verdict (any yes = defective).
CHECKLIST = (
    ("ng", 'Is there an "NG" label, stamp or handwritten mark, or a red stamp with the Japanese characters "不良"?'),
    ("reject", 'Is there a "REJECT" stamp or sticker?'),
    ("tear", "Is the cardboard torn, ripped or punctured anywhere?"),
    ("dent", "Is the box dented, caved in, or crushed at an edge or corner?"),
)
PROMPT_CHECKLIST = ("You are the visual inspector on a cardboard box conveyor line.\n"
                    "Image 1 is the top view of the box and image 2 is the side view.\n"
                    "Answer each question with yes or no, one per line, as '1. yes'.\n"
                    + "\n".join(f"{i + 1}. {q}" for i, (_, q) in enumerate(CHECKLIST)))

PROMPTS = {"verdict": PROMPT, "describe": PROMPT_DESCRIBE, "checklist": PROMPT_CHECKLIST}


def fewshot_prompt(captions: list[str]) -> str:
    """Checklist preceded by labelled example images from this line (few-shot, in context)."""
    n = len(captions)
    return ("You are the visual inspector on a cardboard box conveyor line.\n"
            f"Images 1 to {n} are labelled examples from this line:\n"
            + "".join(f"Image {i + 1}: {c}\n" for i, c in enumerate(captions))
            + f"Image {n + 1} is the top view and image {n + 2} is the side view of the box to inspect now.\n"
            "Answer each question about the box to inspect with yes or no, one per line, as '1. yes'.\n"
            + "\n".join(f"{i + 1}. {q}" for i, (_, q) in enumerate(CHECKLIST)))


def load_exemplars(directory: str | Path) -> list[tuple[Image.Image, str]]:
    """``index.json`` in ``directory``: [{"file": ..., "caption": ...}, ...]."""
    directory = Path(directory)
    return [(Image.open(directory / e["file"]).convert("RGB"), e["caption"])
            for e in json.loads((directory / "index.json").read_text())]
_VERDICT = re.compile(r"\b(DEFECTIVE|OK)\b", re.IGNORECASE)
_ANSWER = re.compile(r"^\s*(\d)\s*[.):]\s*(yes|no)\b", re.IGNORECASE | re.MULTILINE)
_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


class Molmo2Inspector:
    def __init__(self, model_id: str = "allenai/Molmo2-8B", max_new_tokens: int = 96, exemplars=None):
        self.model_id, self.max_new_tokens = model_id, max_new_tokens
        self.exemplars = load_exemplars(exemplars) if exemplars else []
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.bfloat16
        ).to("cuda").eval()

    def judge(self, images: list, style: str = "checklist", top_crop: float | None = 0.5) -> dict:
        """``images``: [top, side] as PIL images or HxWx3 uint8 arrays. Returns verdict, reason, raw text.

        ``top_crop`` keeps that central fraction of the top view: the box always stops under the camera's
        optical axis, and the full frame spends most of its pixels on rollers."""
        images = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im, np.uint8)) for im in images]
        if top_crop:
            w, h = images[0].size
            cw, ch = w * top_crop, h * top_crop
            images[0] = images[0].crop((round((w - cw) / 2), round((h - ch) / 2), round((w + cw) / 2), round((h + ch) / 2)))
        if style == "checklist_fewshot":
            prompt = fewshot_prompt([caption for _, caption in self.exemplars])
            images = [im for im, _ in self.exemplars] + images
        else:
            prompt = PROMPTS[style]
        text = self._generate(prompt, images)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if style.startswith("checklist"):
            answers = {int(n): a.lower() == "yes" for n, a in _ANSWER.findall(text)}
            found = [name for i, (name, _) in enumerate(CHECKLIST) if answers.get(i + 1)]
            verdict = "UNKNOWN" if len(answers) < len(CHECKLIST) else ("DEFECTIVE" if found else "OK")
            reason = ", ".join(found)
        else:
            matches = _VERDICT.findall(text)
            verdict = matches[-1].upper() if style == "describe" and matches else (
                matches[0].upper() if matches else "UNKNOWN")
            reason = lines[0] if style == "describe" else (lines[1] if len(lines) > 1 else "")
        return {"defective": verdict == "DEFECTIVE", "verdict": verdict, "reason": reason, "raw": text}

    def ask(self, images: list, question: str) -> dict:
        """A yes/no question about ``images`` (the line's state checks). ``answer`` is None if unparseable."""
        images = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im, np.uint8)) for im in images]
        text = self._generate(f"{question}\nAnswer with one word: yes or no.", images, max_new_tokens=8)
        match = _YES_NO.search(text)
        return {"answer": None if match is None else match.group(1).lower() == "yes", "raw": text}

    def _generate(self, prompt: str, images: list, max_new_tokens: int | None = None) -> str:
        messages = [{"role": "user", "content": [dict(type="text", text=prompt),
                                                 *(dict(type="image", image=im) for im in images)]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].size(1):], skip_special_tokens=True).strip()

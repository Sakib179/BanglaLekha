"""Experimental Bangla word reconstruction using a saved character classifier.

No fitting, fine-tuning, template training or dictionary correction occurs here.
The returned joined word is an inference result, not a claim of word accuracy.
"""

import json
import math
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF


DEFAULT_SETTINGS = {
    "max_width": 1400,
    "max_segments": 24,
    "min_confidence": 0.30,
    "headline_min_coverage": 0.55,
    "modifier_rules": True,
    "top_k": 3,
}


class ResNetCharacterModel(nn.Module):
    def __init__(self):
        super().__init__()

        network = torchvision.models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(network.children())[:-2])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.25), nn.Linear(512, 84),
        )

    def forward(self, images):
        return self.head(self.encoder(images))


def load_engine(checkpoint, labels_path="labels.json", device="cpu", settings=None):
    # These are the user's own inference weights; load only tensor-safe objects.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)

    if saved.get("architecture") != "resnet18" or saved.get("task") != "character":
        raise ValueError("Expected the saved ResNet-18 character checkpoint.")

    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if set(labels) != {str(i) for i in range(1, 85)}:
        raise ValueError("labels.json must contain the original class IDs 1..84.")

    model = ResNetCharacterModel()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()

    return {
        "model": model, "labels": labels, "device": torch.device(device),
        "height": int(saved["height"]), "width": int(saved["width"]),
        "settings": {**DEFAULT_SETTINGS, **(settings or {})},
        "checkpoint": str(checkpoint), "metadata": {k: v for k, v in saved.items() if k != "state_dict"},
    }


def foreground_gray(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.copy()
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)

    pixels = np.array(ImageOps.exif_transpose(image).convert("L"), dtype=np.uint8)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])

    if np.median(border) > 127:
        pixels = 255 - pixels

    if int(pixels.max()) - int(pixels.min()) < 16:
        raise ValueError("Blank or almost constant image.")

    return pixels


def character_tensor(pixels, height=128, width=128):
    # Match normalise_ink(), image_canvas() and canvas_tensor() from training.
    y, x = np.where(pixels > 32)
    if not len(x):
        raise ValueError("Empty character crop.")

    image = Image.fromarray(pixels[y.min():y.max() + 1, x.min():x.max() + 1])
    scale = min((height - 8) / image.height, (width - 8) / image.width)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.BILINEAR)

    canvas = Image.new("L", (width, height), 0)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    tensor = TF.to_tensor(canvas).expand(3, -1, -1)

    return TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prepare_word(image, settings):
    gray = foreground_gray(image)
    y, x = np.where(gray > 32)
    if not len(x):
        raise ValueError("No visible writing.")

    gray = gray[y.min():y.max() + 1, x.min():x.max() + 1]
    if gray.shape[1] > settings["max_width"]:
        scale = settings["max_width"] / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray, mask


def find_headline(mask, minimum_coverage):
    height, width = mask.shape
    if width < 20 or height < 12:
        return None

    # A strong horizontal run in the upper portion is a candidate matra.
    kernel_width = max(5, min(width // 3, int(height * 0.6)))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, kernel_width), np.uint8))
    counts = np.count_nonzero(horizontal, axis=1)
    end = max(2, int(height * 0.65))
    row = int(np.argmax(counts[:end]))

    if counts[row] / width < minimum_coverage:
        return None

    top = bottom = row
    while top > 0 and counts[top - 1] >= counts[row] * 0.65:
        top -= 1
    while bottom + 1 < end and counts[bottom + 1] >= counts[row] * 0.65:
        bottom += 1

    return top, bottom + 1


def component_boxes(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(2, round(mask.size * 0.00012))
    boxes = []

    for x, y, width, height, area in stats[1:count]:
        if area >= minimum_area:
            boxes.append([int(x), int(y), int(x + width), int(y + height)])

    return sorted(boxes)


def union_box(left, right):
    return [min(left[0], right[0]), min(left[1], right[1]),
            max(left[2], right[2]), max(left[3], right[3])]


def group_vertical_parts(boxes):
    # Keep a detached dot and the body beneath it in one candidate crop.
    groups = []

    for box in sorted(boxes):
        matches = []
        for index, group in enumerate(groups):
            overlap = min(box[2], group[2]) - max(box[0], group[0])
            small_width = min(box[2] - box[0], group[2] - group[0])
            if overlap > max(0, small_width * 0.25):
                matches.append(index)

        for index in reversed(matches):
            box = union_box(box, groups.pop(index))
        groups.append(box)

    return sorted(groups)


def boxes_for_intervals(mask, intervals):
    boxes = []
    for left, right in intervals:
        y, x = np.where(mask[:, left:right] > 0)
        if len(x):
            boxes.append([left + int(x.min()), int(y.min()),
                          left + int(x.max()) + 1, int(y.max()) + 1])

    return boxes


def projection_intervals(mask):
    occupied = np.count_nonzero(mask, axis=0) > 0
    transitions = np.diff(np.pad(occupied.astype(int), (1, 1)))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    return list(zip(starts.tolist(), ends.tolist()))


def split_wide_boxes(mask, boxes):
    # Offer a second hypothesis for touching glyphs instead of always forcing a cut.
    output = []
    for box in boxes:
        left, top, right, bottom = box
        height, width = bottom - top, right - left
        if width < height * 1.35:
            output.append(box)
            continue

        projection = np.count_nonzero(mask[top:bottom, left:right], axis=0)
        low, high = max(1, int(width * 0.30)), min(width - 1, int(width * 0.70))
        if high <= low:
            output.append(box)
            continue
        cut = low + int(np.argmin(projection[low:high]))

        if projection[cut] <= max(2, projection.max() * 0.35):
            output.extend(boxes_for_intervals(mask, [(left, left + cut), (left + cut, right)]))
        else:
            output.append(box)

    return output


def restore_crop_extent(mask, boxes):
    # Segmentation may erase the matra, but inference retains original strokes.
    intervals = [(box[0], box[2]) for box in boxes]

    return boxes_for_intervals(mask, intervals)


def segmentation_candidates(mask, settings):
    candidates = [("components", group_vertical_parts(component_boxes(mask)))]
    headline = find_headline(mask, settings["headline_min_coverage"])
    body = mask.copy()

    if headline is not None:
        body[headline[0]:headline[1]] = 0
        grouped = group_vertical_parts(component_boxes(body))
        candidates.append(("matra-components", restore_crop_extent(mask, grouped)))

    projection = boxes_for_intervals(body, projection_intervals(body))
    candidates.append(("vertical-projection", restore_crop_extent(mask, projection)))
    candidates.append(("touching-valley", restore_crop_extent(mask, split_wide_boxes(body, projection))))

    # Disconnected pieces can belong to one letter, as in an independently written আ.
    if len(candidates[0][1]) <= 8:
        components = candidates[0][1]
        for index in range(len(components) - 1):
            joined = union_box(components[index], components[index + 1])
            if joined[2] - joined[0] < (joined[3] - joined[1]) * 1.45:
                boxes = components[:index] + [joined] + components[index + 2:]
                candidates.append((f"merge-parts-{index + 1}", boxes))

    result, seen = [], set()
    for name, boxes in candidates:
        key = tuple(tuple(box) for box in boxes)
        if key and key not in seen and len(boxes) <= settings["max_segments"]:
            result.append((name, boxes))
            seen.add(key)

    return result, headline, body


def crop_options(gray, boxes, settings):
    options = []
    baseline = float(np.median([box[3] for box in boxes]))
    median_height = float(np.median([box[3] - box[1] for box in boxes]))

    for index, box in enumerate(boxes):
        left, top, right, bottom = box
        crop = gray[top:bottom, left:right]
        options.append({"segment": index, "pixels": crop, "modifier": "", "rule": "classifier"})

        # A descending, narrow extra region can be a short-u sign; this remains heuristic.
        extra = bottom - baseline
        if settings["modifier_rules"] and len(boxes) > 1 and extra > median_height * 0.12:
            cut = int(round(baseline - top))
            lower = crop[cut:]
            if cut > crop.shape[0] * 0.55 and lower.size and np.count_nonzero(lower > 32) > 3:
                y, x = np.where(lower > 32)
                if len(x) and x.max() - x.min() + 1 < crop.shape[1] * 0.85:
                    options.append({"segment": index, "pixels": crop[:cut],
                                    "modifier": "ু", "rule": "lower-zone short-u hypothesis"})

    return options


@torch.inference_mode()
def classify_options(engine, options):
    tensors = [character_tensor(o["pixels"], engine["height"], engine["width"]) for o in options]
    probabilities = []

    for start in range(0, len(tensors), 32):
        batch = torch.stack(tensors[start:start + 32]).to(engine["device"])
        probabilities.extend(engine["model"](batch).softmax(-1).cpu().numpy())

    for option, scores in zip(options, probabilities):
        indices = np.argsort(scores)[-engine["settings"]["top_k"]:][::-1]
        option["top_k"] = [{"class_id": int(i) + 1, "text": engine["labels"][str(int(i) + 1)],
                            "probability": float(scores[i])} for i in indices]
        option["confidence"] = option["top_k"][0]["probability"]
        option["text"] = option["top_k"][0]["text"]
        option["class_id"] = option["top_k"][0]["class_id"]

    return options


def is_aa_stroke(gray, box, median_height):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if height < median_height * 0.65 or width / max(height, 1) > 0.65:
        return False

    foreground = gray[top:bottom, left:right] > 32
    middle = foreground[int(height * 0.25):max(1, int(height * 0.90))]
    spans = []
    for row in middle:
        x = np.flatnonzero(row)
        if len(x):
            spans.append(int(x[-1] - x[0] + 1))

    return (bool(spans) and len(spans) >= len(middle) * 0.75 and
            np.median(spans) <= height * 0.22 and
            np.count_nonzero(foreground, axis=0).max() >= height * 0.60)


def choose_tokens(gray, boxes, options, settings):
    tokens = []
    median_height = float(np.median([b[3] - b[1] for b in boxes]))

    for index, box in enumerate(boxes):
        candidates = [o for o in options if o["segment"] == index]
        # A modifier hypothesis needs a clear classifier improvement over the full crop.
        best = max(candidates, key=lambda o: math.log(max(o["confidence"], 1e-8)) - 0.30 * bool(o["modifier"]))
        token = {k: v for k, v in best.items() if k != "pixels"}
        token["box"] = list(box)

        if settings["modifier_rules"] and tokens and is_aa_stroke(gray, box, median_height):
            if not tokens[-1]["text"].isdigit():
                token.update(text="া", class_id=None, modifier="",
                             rule="right vertical aa-sign hypothesis", confidence=0.55)

        if token["confidence"] < settings["min_confidence"]:
            token["text"] = "�"
        tokens.append(token)

    return tokens


def score_tokens(tokens):
    score = 0.0
    for token in tokens:
        left, top, right, bottom = token["box"]
        ratio = (right - left) / max(1, bottom - top)
        score += math.log(max(token["confidence"], 1e-8))
        score -= max(0.0, ratio - 1.3) * 0.7
        score -= 0.6 * (token["text"] == "�")

    return score / max(len(tokens), 1)


def assemble_word(tokens):
    # Unicode shaping joins the output; no expected answer or dictionary is supplied.
    text = "".join(token["text"] + token["modifier"] for token in tokens)

    return unicodedata.normalize("NFC", text.replace("অা", "আ"))


def prediction_warnings(tokens, headline):
    warnings = []
    if any(token["rule"] != "classifier" for token in tokens):
        warnings.append("Vowel-sign geometry rules were used; inspect the segmentation preview.")
    if any(token["text"] == "�" for token in tokens):
        warnings.append("At least one segment has low classifier confidence.")
    if headline is not None:
        warnings.append("A matra-removal hypothesis was considered; touching characters may remain ambiguous.")

    warnings.append("This prototype has no trained dependent-vowel or word model; other modifiers/conjuncts may fail.")

    return warnings


def predict_word(engine, image):
    started = time.perf_counter()
    gray, mask = prepare_word(image, engine["settings"])
    candidates, headline, body = segmentation_candidates(mask, engine["settings"])
    alternatives = []

    for method, boxes in candidates:
        options = crop_options(gray, boxes, engine["settings"])
        options = classify_options(engine, options)
        tokens = choose_tokens(gray, boxes, options, engine["settings"])
        alternatives.append({"method": method, "word": assemble_word(tokens),
                             "score": score_tokens(tokens), "tokens": tokens})

    if not alternatives:
        raise ValueError("No usable character segmentation. Try clearer writing or increase spacing.")

    alternatives.sort(key=lambda item: item["score"], reverse=True)
    best = alternatives[0]
    confidence = math.exp(sum(math.log(max(t["confidence"], 1e-8)) for t in best["tokens"]) / len(best["tokens"]))

    return {
        "word": best["word"], "method": best["method"], "tokens": best["tokens"],
        "confidence": confidence, "confidence_is_calibrated": False,
        "alternatives": alternatives, "warnings": prediction_warnings(best["tokens"], headline),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "debug": {"gray": gray, "mask": mask, "body_mask": body, "headline": headline},
    }

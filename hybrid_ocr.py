"""Frozen BanglaLekha ResNet roots plus a frozen, separately sourced vowel head.

Inference only. External root predictions are never available to the decoder.
"""

import csv
import hashlib
import json
import math
import os
import time
import unicodedata
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import segmentation_ocr as base_ocr


ROOT = Path(__file__).resolve().parent
HELPER_REVISION = "57263b0706c70033109160dcbb2ea3b382a7d827"
HELPER_URL = "https://raw.githubusercontent.com/ifrit98/bengaliai/" + HELPER_REVISION + "/"
HELPER_FILES = {
    "model.h5": ("model-weights-best-checkpoint.h5", "9403fd7f8189a197f697188b6027ea204551d74b1e30bd1912bf2e4e3f9a7951"),
    "class_map.csv": ("csv/class_map.csv", "2e731b01424d19933778270f59085173ee40427da9a013cf2f2d2bc681ed51be"),
}
SETTINGS = {
    "max_width": 1100, "max_atoms": 20, "max_span": 3,
    "max_regions": 128, "beam_width": 10, "top_k": 3,
    "min_base_score": 0.50, "min_vowel_score": 0.50,
    "min_path_margin": 0.08, "batch_size": 4,
    "vowel_tta": False,
    "min_sign_hypothesis": 0.85,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_vowel_helper(folder=None, download=False):
    """Explicit setup downloads pinned assets once; inference itself stays offline."""
    folder = Path(folder or ROOT / "models/vowel_helper")
    folder.mkdir(parents=True, exist_ok=True)

    for name, (source, expected) in HELPER_FILES.items():
        target = folder / name
        if not target.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {target}. Run prepare_vowel_helper(download=True) once.")
            temporary = target.with_suffix(target.suffix + ".download")
            try:
                with urllib.request.urlopen(HELPER_URL + source, timeout=90) as response:
                    with temporary.open("wb") as handle:
                        while block := response.read(1024 * 1024):
                            handle.write(block)
                if file_sha256(temporary) != expected:
                    raise ValueError(f"Downloaded checksum mismatch: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if file_sha256(target) != expected:
            raise ValueError(f"Checksum mismatch: {target}. Restore the pinned original file.")

    return folder


def clean_legacy_config(node):
    """Translate old serialization fields; every numerical weight stays untouched."""
    if isinstance(node, list):
        return [clean_legacy_config(item) for item in node]
    if not isinstance(node, dict):
        return node
    node = {key: clean_legacy_config(value) for key, value in node.items()}
    kind, config = node.get("class_name"), node.get("config", {})
    if not isinstance(config, dict):
        return node
    if "trainable" in config and config["trainable"] is None:
        config["trainable"] = False  # Old R/Keras merge layers serialized this as null.
    if kind in {"GlorotUniform", "Zeros", "Ones", "VarianceScaling", "RandomNormal", "RandomUniform"}:
        config.pop("dtype", None)  # Initializer metadata; loaded weights replace initial values.
    if kind == "SeparableConv2D":
        for key in ["kernel_initializer", "kernel_regularizer", "kernel_constraint"]:
            config.pop(key, None)  # The separate depthwise/pointwise options are retained.
    if kind == "BatchNormalization":
        if config.get("renorm") or config.get("virtual_batch_size"):
            raise ValueError("Unexpected normalization math in the pinned helper.")
        for key in ["fused", "renorm", "renorm_clipping", "renorm_momentum", "virtual_batch_size", "adjustment"]:
            config.pop(key, None)
        if isinstance(config.get("axis"), list):
            config["axis"] = config["axis"][0]
    return node


def load_vowel_head(folder):
    # Choose the backend before importing Keras. No TensorFlow installation is needed.
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import h5py
    import keras

    if keras.backend.backend() != "torch":
        raise RuntimeError("Restart the kernel and set KERAS_BACKEND=torch before importing Keras.")
    path = Path(folder) / "model.h5"
    with h5py.File(path, "r") as handle:
        config = clean_legacy_config(json.loads(handle.attrs["model_config"]))
    model = keras.models.model_from_json(json.dumps(config), custom_objects={"Model": keras.Model})
    model.load_weights(path)  # Strict loading: no skipped or randomly initialized weights.
    model.trainable = False
    vowel_head = keras.Model(model.inputs[0], model.get_layer("vowel").output)
    vowel_head.trainable = False
    if vowel_head.output_shape[-1] != 11 or vowel_head.trainable_variables:
        raise ValueError("Expected one frozen eleven-class vowel output.")
    return vowel_head


def load_vowel_labels(folder):
    with (Path(folder) / "class_map.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {int(row["label"]): "" if row["component"] == "0" else row["component"]
              for row in rows if row["component_type"] == "vowel_diacritic"}
    if set(labels) != set(range(11)) or labels[0] != "":
        raise ValueError("Unexpected vowel label inventory.")
    return labels


def load_hybrid_engine(checkpoint, labels_path=None, helper_dir=None, settings=None):
    helper_dir = prepare_vowel_helper(helper_dir, download=False)
    resnet = base_ocr.load_engine(checkpoint, labels_path or ROOT / "labels.json", device="cpu")
    for parameter in resnet["model"].parameters():
        parameter.requires_grad_(False)
    return {
        "resnet": resnet, "vowel_head": load_vowel_head(helper_dir),
        "vowel_labels": load_vowel_labels(helper_dir),
        "settings": {**SETTINGS, **(settings or {})},
        "provenance": {"base": "your frozen BanglaLekha ResNet-18",
                       "vowels": "ifrit98/bengaliai", "revision": HELPER_REVISION,
                       "helper_root_output_used": False, "training_performed": False},
    }


def vowel_canvas(pixels):
    """Adapt a crop to the upstream 137x236 source canvas, then its 128px crop."""
    y, x = np.where(pixels > 32)
    if not len(x):
        raise ValueError("Empty grapheme crop.")
    image = Image.fromarray(pixels[y.min():y.max() + 1, x.min():x.max() + 1])
    scale = min(117 / image.height, 216 / image.width)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (236, 137), 0)
    canvas.paste(image, ((236 - image.width) // 2, (137 - image.height) // 2))
    pixels = np.array(canvas)

    # Match the upstream crop thresholds, margins, square padding and cv2 resize.
    y, x = np.where(pixels[5:-5, 5:-5] > 80)
    left, right = max(0, int(x.min()) - 13), min(236, int(x.max()) + 13)
    top, bottom = max(0, int(y.min()) - 10), min(137, int(y.max()) + 10)
    crop = pixels[top:bottom, left:right].copy()
    crop[crop < 28] = 0
    height, width = crop.shape
    side = max(height, width) + 16
    crop = np.pad(crop, (((side - height) // 2,) * 2, ((side - width) // 2,) * 2))
    return cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR).astype(np.float32)


@torch.inference_mode()
def vowel_probabilities(engine, crops):
    canvases, indices, seen = [], [], {}
    for crop in crops:
        canvas = vowel_canvas(crop)
        key = canvas.tobytes()
        if key not in seen:
            seen[key] = len(canvases)
            canvases.append(canvas)
        indices.append(seen[key])
    if not canvases:
        return np.empty((0, 11), dtype=np.float32)
    batch_size = engine["settings"]["batch_size"]
    scores = []
    for start in range(0, len(canvases), batch_size):
        batch = np.stack(canvases[start:start + batch_size])
        mean = batch.mean(axis=(1, 2), keepdims=True)
        std = np.maximum(batch.std(axis=(1, 2), keepdims=True), 1e-6)
        normal = (batch - mean) / std
        first = engine["vowel_head"](normal[..., None], training=False).detach().cpu().numpy()
        if engine["settings"]["vowel_tta"]:
            # Optional second input convention from the upstream augmentation generator.
            scaled = normal / np.maximum(normal.max(axis=(1, 2), keepdims=True), 1e-6)
            second = engine["vowel_head"](scaled[..., None], training=False).detach().cpu().numpy()
            first = (first + second) * 0.5
        scores.extend(first)
    return np.asarray(scores)[indices]


def adaptive_body(mask):
    """Follow a slightly curved matra; use only the middle zone to propose cuts."""
    height, width = mask.shape
    if height < 16 or width < 20:
        return None
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                  np.ones((1, max(5, round(height * 0.25))), np.uint8))
    horizontal[round(height * 0.55):] = 0
    if np.count_nonzero(np.any(horizontal, axis=0)) < width * 0.35:
        return None
    seed = int(np.count_nonzero(horizontal, axis=1).argmax())
    centers, thickness = np.full(width, np.nan), []
    for x in range(width):
        ys = np.flatnonzero(horizontal[:, x])
        ys = ys[np.abs(ys - seed) <= max(3, height * 0.15)]
        if not len(ys):
            continue
        y = int(ys[np.abs(ys - seed).argmin()])
        top = bottom = y
        while top and horizontal[top - 1, x]:
            top -= 1
        while bottom + 1 < height and horizontal[bottom + 1, x]:
            bottom += 1
        centers[x] = (top + bottom) / 2
        thickness.append(bottom - top + 1)
    known = np.flatnonzero(np.isfinite(centers))
    if not len(known):
        return None
    centers = np.interp(np.arange(width), known, centers[known])
    radius = max(2, round(float(np.median(thickness)) * 0.75))
    bottoms = [int(np.flatnonzero(mask[:, x])[-1]) for x in range(width) if np.any(mask[:, x])]
    baseline = int(np.percentile(bottoms, 40)) + 1
    body, middle = mask.copy(), mask.copy()
    for x, center in enumerate(centers):
        top, bottom = max(0, round(center) - radius), min(height, round(center) + radius + 1)
        body[top:bottom, x] = 0
        middle[:bottom, x] = 0
    middle[baseline:] = 0
    return body, middle, baseline


def grapheme_regions(gray, mask, settings):
    options = {**base_ocr.DEFAULT_SETTINGS, **settings}
    proposals, headline, body = base_ocr.segmentation_candidates(mask, options)
    adaptive = adaptive_body(mask)
    baseline = None
    if adaptive is not None:
        curved, middle, baseline = adaptive
        grouped = base_ocr.group_vertical_parts(base_ocr.component_boxes(curved))
        projected = base_ocr.boxes_for_intervals(middle, base_ocr.projection_intervals(middle))
        for name, boxes in [("adaptive-matra", grouped), ("middle-zone", projected)]:
            if boxes:
                proposals.append((name, base_ocr.restore_crop_extent(mask, boxes)))
    regions, graphs = [], []
    for method, boxes in proposals:
        if method.startswith("merge-parts") or len(boxes) > settings["max_atoms"]:
            continue
        graph = len(graphs)
        graphs.append({"method": method, "boxes": boxes})
        for start in range(len(boxes)):
            for end in range(start + 1, min(len(boxes), start + settings["max_span"]) + 1):
                box = boxes[start]
                for following in boxes[start + 1:end]:
                    box = base_ocr.union_box(box, following)
                left, top, right, bottom = box
                if (right - left) / max(1, bottom - top) <= 2.8:
                    regions.append({"start": start, "end": end, "box": box,
                                    "graph": graph, "graph_size": len(boxes), "method": method,
                                    "baseline": baseline})
    if not regions:
        raise ValueError("No usable writing. Try a clean single-word crop.")
    if len(regions) > settings["max_regions"]:
        raise ValueError("Candidate limit exceeded. Use a shorter or clearer word crop.")
    return regions, graphs, body, headline


def root_views(pixels, sign):
    """Conditional crop proposals; these are candidates, never irreversible edits."""
    height, width = pixels.shape
    views = [("full", (0, 0, width, height), 0.10 if sign else 0.0)]
    if not sign:
        return views
    if sign in "ুূৃ":
        parts = base_ocr.component_boxes(np.where(pixels > 80, 255, 0).astype(np.uint8))
        if parts:
            largest = max(parts, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            if height * 0.55 < largest[3] < height * 0.94:
                views.append(("detached lower-sign crop", (0, 0, width, largest[3]), 0.03))
                return views  # Use the observed component boundary before cutting through ink.
    fractions = [0.10, 0.20, 0.30, 0.35, 0.40]
    if sign in "ুূৃ":
        fractions += [0.45, 0.50, 0.55, 0.60]  # A lower sign can be taller than half the base.
    for fraction in fractions:
        dx, dy = max(1, round(width * fraction)), max(1, round(height * fraction))
        if sign in "াী":
            box = (0, 0, width - dx, height)
        elif sign in "িেৈ":
            box = (dx, 0, width, height)
        elif sign in "ুূৃ":
            box = (0, 0, width, height - dy)
        else:  # ও/ঔ signs occupy both sides of the base.
            box = (max(1, dx // 2), 0, width - max(1, dx // 2), height)
        left, top, right, bottom = box
        if right > left and bottom > top and np.any(pixels[top:bottom, left:right] > 32):
            views.append(("sign-aware crop", box, fraction * 0.20))
    return views


def root_jobs(gray, regions, vowel_scores, engine):
    jobs, seen = [], {}
    settings = engine["settings"]
    for region, scores in zip(regions, vowel_scores):
        left, top, right, bottom = region["box"]
        pixels = gray[top:bottom, left:right]
        vowel_ids = scores.argsort()[-settings["top_k"]:][::-1].tolist()
        # Keep the no-sign hypothesis available for independent vowels and digits.
        if 0 not in vowel_ids:
            vowel_ids.append(0)
        vowel_ids = [i for i in vowel_ids if i == 0 or scores[i] >= settings["min_sign_hypothesis"]]
        region["vowel_ids"] = vowel_ids
        region["views"] = []
        for vowel_id in vowel_ids:
            sign = engine["vowel_labels"][vowel_id]
            views = root_views(pixels, sign)
            if sign in {"ু", "ূ", "ৃ"} and region.get("baseline") is not None:
                cut = region["baseline"] - top
                for offset in [-2, 0, 2]:
                    bottom_cut = cut + offset
                    if pixels.shape[0] * 0.35 < bottom_cut < pixels.shape[0] * 0.95:
                        views.append(("word-baseline crop", (0, 0, pixels.shape[1], bottom_cut), 0.03))
            if sign in {"ু", "ূ", "ৃ"}:
                views = refine_lower_views(views)
            for name, box, penalty in views:
                x1, y1, x2, y2 = box
                crop = pixels[y1:y2, x1:x2]
                key = (crop.shape, hashlib.sha256(crop.tobytes()).digest())
                if key not in seen:
                    seen[key] = len(jobs)
                    jobs.append({"pixels": crop})
                region["views"].append({"job": seen[key], "vowel_id": vowel_id,
                                        "box": [left + x1, top + y1, left + x2, top + y2],
                                        "view": name, "penalty": penalty})
    return jobs


def refine_lower_views(views):
    """Small horizontal insets remove neighboring strokes from a lower-sign crop."""
    refined = list(views)
    for name, box, penalty in views:
        if name == "full":
            continue
        left, top, right, bottom = box
        width, height = right - left, bottom - top
        step = max(1, round(min(width, height) * 0.07))
        for trim_left in [0, step, 2 * step]:
            for trim_right in [0, step, 2 * step]:
                if not (trim_left or trim_right) or trim_left + trim_right > width * 0.4:
                    continue
                proposed = (left + trim_left, top, right - trim_right, bottom)
                extra = 0.10 * (trim_left + trim_right) / width
                refined.append((name + " + boundary refinement", proposed, penalty + extra))
    return refined


def valid_attachment(base, sign):
    if not sign:
        return True
    if base == "অ" and sign == "া":
        return True  # Orthographic spelling of independent আ.
    return bool(base) and ("ক" <= base[0] <= "হ" or base[0] in "ড়ঢ়য়")


def region_tokens(region, jobs, scores, engine):
    choices = {}
    for view in region["views"]:
        vowel_id = view["vowel_id"]
        sign = engine["vowel_labels"][vowel_id]
        if sign and scores[vowel_id] < engine["settings"]["min_sign_hypothesis"]:
            continue  # Weak sign evidence remains a warning, rather than an invented extra vowel.
        for root in jobs[view["job"]]["top_k"]:
            if not valid_attachment(root["text"], sign):
                continue
            left, top, right, bottom = view["box"]
            ratio = (right - left) / max(1, bottom - top)
            penalty = view["penalty"] + 0.16 * (region["end"] - region["start"] - 1)
            penalty += 0.7 * max(0.0, ratio - 1.25)
            penalty += 1.0 * (ratio < 0.15)
            if base_ocr.is_aa_stroke(jobs[view["job"]]["pixels"],
                                     [0, 0, right - left, bottom - top], bottom - top):
                penalty += 1.5  # A bare vertical sign is weak evidence for a complete base.
            score = 0.5 * (math.log(max(root["probability"], 1e-8)) +
                           math.log(max(float(scores[vowel_id]), 1e-8))) - penalty
            token = {"base": root["text"], "base_class_id": root["class_id"],
                     "vowel": sign, "vowel_class_id": vowel_id,
                     "base_probability": root["probability"], "vowel_probability": float(scores[vowel_id]),
                     "box": region["box"], "base_box": view["box"], "view": view["view"],
                     "score": score, "base_source": "trained_resnet18",
                     "base_top_k": jobs[view["job"]]["top_k"]}
            key = (root["class_id"], vowel_id)
            if key not in choices or score > choices[key]["score"]:
                choices[key] = token
    return sorted(choices.values(), key=lambda token: token["score"], reverse=True)[:10]


def join_tokens(tokens):
    text = "".join(token["base"] + token["vowel"] for token in tokens)
    return unicodedata.normalize("NFC", text.replace("অা", "আ"))


def decode_graph(regions, atom_count, beam_width):
    # Dynamic programming permits alternatives that merge disconnected sign pieces.
    beams = {0: [(0.0, [])]}
    for position in range(atom_count):
        for total, tokens in beams.get(position, []):
            for region in regions:
                if region["start"] != position:
                    continue
                end = region["end"]
                for token in region["tokens"]:
                    beams.setdefault(end, []).append((total + token["score"], tokens + [token]))
                beams[end].sort(key=lambda path: path[0], reverse=True)
                beams[end] = beams[end][:beam_width]
    paths = {}
    for total, tokens in beams.get(atom_count, []):
        word = join_tokens(tokens)
        score = total
        if word not in paths or score > paths[word]["score"]:
            paths[word] = {"word": word, "score": score, "tokens": tokens}
    return sorted(paths.values(), key=lambda path: path["score"], reverse=True)


def decode_regions(regions, beam_width):
    paths = {}
    for graph in sorted({region["graph"] for region in regions}):
        subset = [region for region in regions if region["graph"] == graph]
        for path in decode_graph(subset, subset[0]["graph_size"], beam_width):
            word = path["word"]
            path["method"] = subset[0]["method"]
            if word not in paths or path["score"] > paths[word]["score"]:
                paths[word] = path
    return sorted(paths.values(), key=lambda path: path["score"], reverse=True)[:beam_width]


def summarize_prediction(paths, engine):
    if not paths:
        raise ValueError("No complete recognition path. Try clearer writing or spacing.")
    settings = engine["settings"]
    for path in paths:
        path["meets_evidence_thresholds"] = all(
            t["base_probability"] >= settings["min_base_score"] and
            t["vowel_probability"] >= settings["min_vowel_score"] for t in path["tokens"])
    eligible = [path for path in paths if path["meets_evidence_thresholds"]]
    ordered = eligible + [path for path in paths if not path["meets_evidence_thresholds"]]
    best = ordered[0]
    competitors = eligible if eligible else paths
    margin = best["score"] - competitors[1]["score"] if len(competitors) > 1 else None
    low_base = any(t["base_probability"] < settings["min_base_score"] for t in best["tokens"])
    low_vowel = any(t["vowel_probability"] < settings["min_vowel_score"] for t in best["tokens"])
    ambiguous = margin is not None and margin < settings["min_path_margin"]
    accepted = not (low_base or low_vowel or ambiguous)
    warnings = ["Scores are uncalibrated; a confident prediction may still be wrong."]
    if low_base:
        warnings.append("At least one base character has weak ResNet evidence.")
    if low_vowel:
        warnings.append("At least one vowel-sign decision has weak evidence.")
    if ambiguous:
        warnings.append("Several different joined words have similar scores.")
    return {"word": best["word"], "accepted_word": best["word"] if accepted else None,
            "status": "accepted" if accepted else "uncertain", "tokens": best["tokens"],
            "score": best["score"], "path_margin": margin, "alternatives": ordered[:5],
            "confidence_is_calibrated": False, "warnings": warnings,
            "provenance": engine["provenance"]}


def predict_word(engine, image):
    started = time.perf_counter()
    if not engine.get("resnet") or engine["resnet"].get("model") is None:
        raise ValueError("Your trained ResNet is required. No external word-recognizer fallback exists.")
    gray, mask = base_ocr.prepare_word(image, engine["settings"])
    regions, atoms, body, headline = grapheme_regions(gray, mask, engine["settings"])
    crops = [gray[t:b, l:r] for l, t, r, b in [region["box"] for region in regions]]
    vowel_scores = vowel_probabilities(engine, crops)
    jobs = root_jobs(gray, regions, vowel_scores, engine)
    # Every base choice below comes from the user's actual saved classifier.
    jobs = base_ocr.classify_options(engine["resnet"], jobs)
    for region, scores in zip(regions, vowel_scores):
        region["tokens"] = region_tokens(region, jobs, scores, engine)
    paths = decode_regions(regions, engine["settings"]["beam_width"])
    result = summarize_prediction(paths, engine)
    method = result["alternatives"][0]["method"]
    if method in {"adaptive-matra", "middle-zone"}:
        adaptive = adaptive_body(mask)
        body = adaptive[0 if method == "adaptive-matra" else 1]
    result["method"] = method
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    result["candidate_regions"] = len(regions)
    result["base_crops_evaluated"] = len(jobs)
    result["debug"] = {"gray": gray, "mask": mask, "body_mask": body, "headline": headline,
                       "atoms": atoms, "regions": regions, "vowel_scores": vowel_scores}
    return result

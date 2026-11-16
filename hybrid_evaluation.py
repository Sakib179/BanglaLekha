"""Word and vowel diagnostics. Expected text is used only after prediction."""

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import regex

import hybrid_ocr
import segmentation_ocr


SIGNS = "ািীুূৃেৈোৌ"
SIGN_NAMES = {"": "none", **dict(zip(SIGNS, ["aa", "i", "ii", "u", "uu", "ri", "e", "ai", "o", "au"]))}


def align_sequences(expected, predicted):
    """Levenshtein alignment; None denotes an insertion or deletion."""
    rows, cols = len(expected) + 1, len(predicted) + 1
    table = np.zeros((rows, cols), dtype=int)
    table[:, 0], table[0, :] = np.arange(rows), np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            table[i, j] = min(table[i - 1, j] + 1, table[i, j - 1] + 1,
                              table[i - 1, j - 1] + (expected[i - 1] != predicted[j - 1]))
    pairs, i, j = [], rows - 1, cols - 1
    while i or j:
        if i and j and table[i, j] == table[i - 1, j - 1] + (expected[i - 1] != predicted[j - 1]):
            pairs.append((expected[i - 1], predicted[j - 1]))
            i, j = i - 1, j - 1
        elif i and table[i, j] == table[i - 1, j] + 1:
            pairs.append((expected[i - 1], None))
            i -= 1
        else:
            pairs.append((None, predicted[j - 1]))
            j -= 1
    return list(reversed(pairs)), int(table[-1, -1])


def vowel_rows(expected, predicted):
    expected = regex.findall(r"\X", unicodedata.normalize("NFC", expected))
    predicted = regex.findall(r"\X", unicodedata.normalize("NFC", predicted))
    aligned, _ = align_sequences(expected, predicted)
    rows = []
    for truth, guess in aligned:
        left = "missing" if truth is None else "".join(c for c in truth if c in SIGNS)
        right = "missing" if guess is None else "".join(c for c in guess if c in SIGNS)
        rows.append({"expected_grapheme": truth or "", "predicted_grapheme": guess or "",
                     "expected_sign": left, "predicted_sign": right,
                     "sign_correct": left == right,
                     "base_correct": (truth is not None and guess is not None and
                                      "".join(c for c in truth if c not in SIGNS) ==
                                      "".join(c for c in guess if c not in SIGNS))})
    return rows


def save_prediction(result, path):
    record = {key: value for key, value in result.items() if key != "debug"}
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def cached_prediction(engine, image, cache=None):
    """Reuse unchanged notebook inputs; this never changes the recognition algorithm."""
    if cache is None:
        return hybrid_ocr.predict_word(engine, image)
    key = (hybrid_ocr.file_sha256(image), id(engine['resnet']['model']), id(engine['vowel_head']),
           json.dumps(engine['settings'], sort_keys=True))
    if key not in cache:
        cache[key] = hybrid_ocr.predict_word(engine, image)
    return cache[key]


def _predict_record(engine, image, cache=None):
    try:
        result = cached_prediction(engine, image, cache)
        return result, ""
    except (OSError, ValueError) as error:
        return {"word": "", "status": "error", "latency_ms": None}, str(error)


def evaluate_words(engine, csv_path, report_dir, evaluation_kind="development", prediction_cache=None):
    csv_path, report_dir = Path(csv_path), Path(report_dir)
    tests = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if not {"path", "expected"}.issubset(tests.columns) or tests.empty:
        raise ValueError("Use a nonempty UTF-8 CSV with path,expected columns.")
    if tests["expected"].str.strip().eq("").any():
        raise ValueError("Every test image needs a reviewed transcription.")
    report_dir.mkdir(parents=True, exist_ok=True)
    records, signs = [], []
    for index, row in enumerate(tests.itertuples(index=False)):
        image = csv_path.parent / row.path
        expected = unicodedata.normalize("NFC", row.expected.strip())
        result, error = _predict_record(engine, image, prediction_cache)
        try:
            baseline = segmentation_ocr.predict_word(engine["resnet"], image)["word"]
        except (OSError, ValueError):
            baseline = ""
        _, edits = align_sequences(expected, result["word"])
        records.append({"path": row.path, "expected": expected, "prediction": result["word"],
                        "status": result["status"], "raw_correct": result["word"] == expected,
                        "accepted_correct": result["status"] == "accepted" and result["word"] == expected,
                        "baseline_prediction": baseline, "baseline_correct": baseline == expected,
                        "edits": edits, "reference_codepoints": len(expected),
                        "latency_ms": result["latency_ms"], "error": error})
        signs.extend({"image": row.path, **item} for item in vowel_rows(expected, result["word"]))
        save_prediction(result, report_dir / f"prediction_{index:03d}.json")
    return write_evaluation(records, signs, report_dir, evaluation_kind)


def write_evaluation(records, signs, report_dir, kind):
    words, vowels = pd.DataFrame(records), pd.DataFrame(signs)
    latency = words["latency_ms"].dropna()
    summary = {"evaluation_kind": kind, "examples": len(words),
               "raw_word_accuracy": float(words["raw_correct"].mean()),
               "accepted_word_accuracy_rejections_as_errors": float(words["accepted_correct"].mean()),
               "acceptance_rate": float(words["status"].eq("accepted").mean()),
               "baseline_word_accuracy": float(words["baseline_correct"].mean()),
               "codepoint_cer": float(words["edits"].sum() / words["reference_codepoints"].sum()),
               "aligned_vowel_error_rate": float(1 - vowels["sign_correct"].mean()),
               "median_latency_ms": float(latency.median()) if len(latency) else None,
               "p95_latency_ms": float(latency.quantile(.95)) if len(latency) else None,
               "training_performed": False}
    words.to_csv(report_dir / "word_predictions.csv", index=False)
    vowels.to_csv(report_dir / "vowel_alignment.csv", index=False)
    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return words, vowels, summary

"""Reproducible real-word sampling, resumable inference, and honest exact accuracy."""

import hashlib
import html
import json
import os
import random
import time
import unicodedata
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import hybrid_ocr
import segmentation_ocr
from hybrid_evaluation import align_sequences


DEVELOPMENT_IDS = {'0', '10', '1128', '1177', '1100', '10911',
                   '1971', '1901', '867', '1013', '20052', '20588'}


def atomic_json(data, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _pixel_hash(path):
    pixels = segmentation_ocr.foreground_gray(path)
    y, x = np.where(pixels > 32)
    if not len(x):
        raise ValueError('Blank sample')
    crop = pixels[y.min():y.max() + 1, x.min():x.max() + 1]
    return hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest()


def prepare_word_sample(dataset_dir, labels_csv, count=200, seed=20261117):
    """Freeze a sample before recognition; never select images by prediction quality."""
    dataset_dir, labels_csv = Path(dataset_dir).resolve(), Path(labels_csv).resolve()
    if labels_csv.exists():
        return read_references(labels_csv, count)
    images = dataset_dir / 'Word Images(bmp)'
    annotations = dataset_dir / 'Word Annotations(xml)'
    paths = sorted(images.glob('*.bmp'), key=lambda p: p.name)
    random.Random(seed).shuffle(paths)
    rows, seen = [], set()
    for path in paths:
        if path.stem in DEVELOPMENT_IDS:
            continue
        try:
            names = [int(obj.findtext('name')) for obj in
                     ET.parse(annotations / f'{path.stem}.xml').findall('object')]
            # Exclude single-component images and pure digit/punctuation examples.
            has_letter = any(1 <= code <= 46 or 67 <= code <= 85 or code == 91 for code in names)
            if len(names) < 2 or not has_letter:
                continue
            pixel_hash = _pixel_hash(path)
        except (OSError, ValueError, ET.ParseError):
            continue
        if pixel_hash in seen:
            continue
        seen.add(pixel_hash)
        rows.append({'test_id': len(rows) + 1,
                     'path': Path(os.path.relpath(path, labels_csv.parent)).as_posix(),
                     'expected': '', 'reference_verified': False, 'source_id': path.stem,
                     'image_sha256': hybrid_ocr.file_sha256(path), 'pixel_sha256': pixel_hash})
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f'Only {len(rows)} eligible word images found; need {count}.')
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(labels_csv, index=False, encoding='utf-8')
    atomic_json({'count': count, 'seed': seed, 'source': str(dataset_dir),
                 'protocol': 'Multi-component word images; exclude known development images, digits-only inputs and exact cropped-pixel duplicates.',
                 'references': 'Blank pending independent transcription; XML IDs are not Unicode transcripts.'},
                labels_csv.with_suffix('.sampling.json'))
    return table


def read_references(csv_path, count=200):
    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if not {'path', 'expected'}.issubset(table.columns) or len(table) != count:
        raise ValueError(f'Reference CSV must contain exactly {count} rows and path,expected columns.')
    if not table['path'].is_unique:
        raise ValueError('Every test image path must be unique.')
    if 'test_id' not in table:
        table['test_id'] = np.arange(1, count + 1)
    if not table['test_id'].astype(str).is_unique:
        raise ValueError('Test IDs must be unique.')
    return table


def write_review_gallery(labels_csv, output_dir):
    """Show source pixels and sample IDs only, so predictions cannot supply labels."""
    labels_csv, output_dir = Path(labels_csv), Path(output_dir)
    table = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    image_dir = output_dir / 'review_images'
    image_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in table.itertuples(index=False):
        source = labels_csv.parent / row.path
        name = f'word_{int(row.test_id):03d}.png'
        with Image.open(source) as image:
            ImageOps.exif_transpose(image).convert('RGB').save(image_dir / name)
        cards.append(f'<figure><figcaption>{html.escape(str(row.test_id))}: '
                     f'{html.escape(source.name)}</figcaption><img src="review_images/{name}" '
                     f'alt="Word {html.escape(str(row.test_id))}"></figure>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<title>200-word reference review</title><style>'
                'body{font:16px sans-serif;margin:24px;background:#eee}'
                'main{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:16px}'
                'figure{background:white;margin:0;padding:16px}img{max-width:100%;height:130px;object-fit:contain}'
                '</style><h1>Reference review: source images only</h1>'
                '<p>Read each word independently. Fill expected and mark reference_verified=true '
                'in the reference CSV. No model predictions are shown here.</p><main>' +
                ''.join(cards) + '</main></html>')
    target = output_dir / 'reference_review.html'
    target.write_text(document, encoding='utf-8')
    return target


def inference_signature(engine):
    context = {'base_checkpoint': hybrid_ocr.file_sha256(engine['resnet']['checkpoint']),
               'vowel_checkpoint': hybrid_ocr.HELPER_FILES['model.h5'][1],
               'settings': engine['settings'],
               'source': {p.name: hybrid_ocr.file_sha256(p) for p in
                          [Path(hybrid_ocr.__file__), Path(segmentation_ocr.__file__)]},
               'versions': {name: version(name) for name in ['torch', 'torchvision', 'keras', 'numpy']}}
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def _run_image(engine, image):
    started = time.perf_counter()
    try:
        result = hybrid_ocr.predict_word(engine, image)
        record = {key: value for key, value in result.items() if key != 'debug'}
        record['error'] = ''
    except (OSError, ValueError, RuntimeError) as error:
        record = {'word': '', 'accepted_word': None, 'status': 'error',
                  'error': f'{type(error).__name__}: {error}',
                  'latency_ms': (time.perf_counter() - started) * 1000}
    return record


def run_word_benchmark(engine, labels_csv, output_dir, count=200, progress=None):
    """Save each prediction atomically. Reference edits never require rerunning inference."""
    labels_csv, output_dir = Path(labels_csv).resolve(), Path(output_dir)
    table = read_references(labels_csv, count)
    cache_dir = output_dir / 'predictions'
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = inference_signature(engine)
    records, reused = [], 0
    for row in table.itertuples(index=False):
        image = labels_csv.parent / row.path
        image_hash = hybrid_ocr.file_sha256(image) if image.is_file() else None
        expected_hash = getattr(row, 'image_sha256', '')
        if expected_hash and image_hash and image_hash != expected_hash:
            raise ValueError(f'Test image changed: {image}. Use a new sample manifest.')
        stamp = {'signature': signature, 'image_sha256': image_hash}
        destination = cache_dir / f'{int(row.test_id):03d}.json'
        cached = json.loads(destination.read_text()) if destination.exists() else None
        if cached and cached.get('cache_stamp') == stamp:
            result, reused = cached['result'], reused + 1
        else:
            result = _run_image(engine, image)
            atomic_json({'cache_stamp': stamp, 'result': result}, destination)
        records.append({'test_id': str(row.test_id), 'path': row.path,
                        'prediction': result['word'], 'status': result['status'],
                        'latency_ms': result['latency_ms'], 'error': result.get('error', '')})
        if progress:
            progress(len(records), count, reused)
        if len(records) % 10 == 0 or len(records) == count:
            atomic_json({'processed': len(records), 'total': count, 'reused': reused,
                         'inference_signature': signature}, output_dir / 'progress.json')
            pd.DataFrame(records).to_csv(output_dir / 'predictions.csv', index=False)
    return score_word_benchmark(labels_csv, output_dir, count=count)


def score_word_benchmark(labels_csv, output_dir, count=200):
    """Main exact accuracy is correct joined words / all requested images, including errors."""
    output_dir = Path(output_dir)
    refs = read_references(labels_csv, count)
    predictions = pd.read_csv(output_dir / 'predictions.csv', dtype={'test_id': str}, keep_default_na=False)
    refs['test_id'] = refs['test_id'].astype(str)
    if len(predictions) != count or not predictions['test_id'].is_unique:
        raise ValueError(f'Complete all {count} predictions before computing final metrics.')
    results = refs.merge(predictions.drop(columns='path'), on='test_id', how='left', validate='one_to_one')
    if results['status'].isna().any():
        raise ValueError('Prediction IDs do not match the reference sample.')
    results['expected'] = results['expected'].map(lambda s: unicodedata.normalize('NFC', str(s).strip()))
    results['prediction'] = results['prediction'].map(lambda s: unicodedata.normalize('NFC', str(s)))
    labeled = results['expected'].str.len().gt(0)
    reviewed = results.get('reference_verified', pd.Series('true', index=results.index))
    reviewed = reviewed.astype(str).str.lower().isin(['true', '1', 'yes']) & labeled
    ready = bool(reviewed.all())
    results['correct'] = results['prediction'].eq(results['expected']).where(reviewed, pd.NA)
    results['accepted_correct'] = (results['correct'] & results['status'].eq('accepted')).where(reviewed, pd.NA)
    correct = int(results['correct'].sum()) if ready else None
    latency = pd.to_numeric(results['latency_ms'], errors='coerce')
    summary = {'total_images': count, 'labeled_images': int(labeled.sum()),
               'verified_reference_images': int(reviewed.sum()),
               'accuracy_status': 'verified_references_complete' if ready else 'awaiting_verified_references',
               'correct_predictions': correct, 'incorrect_predictions': count - correct if ready else None,
               'exact_word_accuracy': correct / count if ready else None,
               'accepted_word_accuracy_rejections_as_errors': float(results['accepted_correct'].sum() / count) if ready else None,
               'inference_errors': int(results['status'].eq('error').sum()),
               'uncertain_predictions': int(results['status'].eq('uncertain').sum()),
               'accepted_predictions': int(results['status'].eq('accepted').sum()),
               'median_latency_ms': float(latency.median()), 'p95_latency_ms': float(latency.quantile(.95)),
               'accuracy_definition': f'NFC-normalized whole-word exact matches / all {count} images; no character averaging or accepted-only denominator.'}
    if ready:
        edits = sum(align_sequences(a, b)[1] for a, b in zip(results['expected'], results['prediction']))
        summary['codepoint_cer'] = edits / results['expected'].str.len().sum()
    results.to_csv(output_dir / 'word_results.csv', index=False)
    atomic_json(summary, output_dir / 'metrics.json')
    return results, summary

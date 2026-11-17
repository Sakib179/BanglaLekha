"""Run bounded CPU batches so long notebook tests can resume after termination."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'examples/promitolipi_200_reference_words.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/hybrid_word_recognition/test_200_words')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'training_runs/macos_resnet18_efficientnetb0_v1/character/resnet18/best_model.pt')
    parser.add_argument('--settings-json', default='{}')
    parser.add_argument('--batch-images', type=int, default=5)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def worker(args):
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    os.environ.setdefault('ONEDNN_PRIMITIVE_CACHE_CAPACITY', '4')
    import torch
    import hybrid_ocr
    from word_benchmark import run_word_benchmark

    torch.set_num_threads(2)
    engine = hybrid_ocr.load_hybrid_engine(args.checkpoint, settings=json.loads(args.settings_json))
    _, summary = run_word_benchmark(engine, args.labels, args.output, count=200,
                                    max_new_predictions=args.batch_images)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def command(args, batch_size):
    return [sys.executable, str(Path(__file__).resolve()), '--worker',
            '--labels', str(args.labels), '--output', str(args.output),
            '--checkpoint', str(args.checkpoint), '--settings-json', args.settings_json,
            '--batch-images', str(batch_size)]


def record_terminated_image(args):
    """A repeatedly terminated single-image attempt is an error, never a dropped sample."""
    with args.labels.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    progress = json.loads((args.output / 'progress.json').read_text())
    for row in rows:
        path = args.output / 'predictions' / f'{int(row["test_id"]):03d}.json'
        if path.exists():
            continue
        image = args.labels.parent / row['path']
        digest = hashlib.sha256(image.read_bytes()).hexdigest() if image.exists() else None
        record = {'cache_stamp': {'signature': progress['inference_signature'], 'image_sha256': digest},
                  'result': {'word': '', 'accepted_word': None, 'status': 'error',
                             'latency_ms': None, 'error': 'Inference worker was terminated on two consecutive attempts, including an isolated single-image attempt.'}}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)
        print(f'Recorded worker-termination error for sample {row["test_id"]}.', flush=True)
        return
    raise RuntimeError('Worker failed without an identifiable missing prediction; inspect its error output.')


def main():
    args = arguments()
    if args.worker:
        worker(args)
        return
    if not args.labels.is_file():
        raise FileNotFoundError('Prepare the fixed 200-image reference CSV in notebook 04 first.')
    while True:
        result = subprocess.run(command(args, args.batch_images), cwd=ROOT)
        if result.returncode:
            if result.returncode not in [-9, 137]:
                raise SystemExit(result.returncode)
            print('Worker terminated; retrying the next image in isolation.', flush=True)
            isolated = subprocess.run(command(args, 1), cwd=ROOT)
            if isolated.returncode in [-9, 137]:
                record_terminated_image(args)
            elif isolated.returncode:
                raise SystemExit(isolated.returncode)
        progress_path = args.output / 'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        print(f"Saved {progress.get('processed', 0)}/200 predictions.", flush=True)
        if progress.get('processed') == 200 and (args.output / 'metrics.json').exists():
            break


if __name__ == '__main__':
    main()

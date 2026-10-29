"""Measure the actual headline-extraction entry point on labelled images.

Accepts a JSON/JSONL list of {image, text} records, or a frozen training
manifest with --split validation/test. Does not train or change model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import ocr_service
from scripts.train_sinhala_ocr import load_benchmark, metrics, normalize, parse_psm_modes, sha256, write_json


@contextmanager
def use_tessdata(directory):
    previous = os.environ.get('SINHALA_TESSDATA_DIR')
    previous_threads = os.environ.get('OMP_THREAD_LIMIT')
    try:
        # Each worker launches Tesseract; avoid CPU oversubscription.
        os.environ.setdefault('OMP_THREAD_LIMIT', '1')
        if directory:
            path = Path(directory).resolve()
            if not all((path / name).is_file() for name in ('sin.traineddata', 'eng.traineddata')):
                raise ValueError('Evaluation requires both sin.traineddata and eng.traineddata')
            os.environ['SINHALA_TESSDATA_DIR'] = str(path)
        yield
    finally:
        if previous is None:
            os.environ.pop('SINHALA_TESSDATA_DIR', None)
        else:
            os.environ['SINHALA_TESSDATA_DIR'] = previous
        if previous_threads is None:
            os.environ.pop('OMP_THREAD_LIMIT', None)
        else:
            os.environ['OMP_THREAD_LIMIT'] = previous_threads


def evaluate_records(records, workers=2, psm_modes=None):
    def recognize(row):
        try:
            result = ocr_service.extract_text(row['image'], psm_modes=psm_modes)
        except Exception as error:
            result = {'status': 'error', 'error': str(error), 'extracted_text': ''}
        return {'image': row['image'], 'reference': row['text'],
                'prediction': result.get('extracted_text', ''),
                'exact_match': normalize(row['text']) == normalize(result.get('extracted_text', '')),
                'result': result}

    predictions = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, prediction in enumerate(pool.map(recognize, records), 1):
            predictions.append(prediction)
            if index % 10 == 0 or index == len(records):
                print(f'Runtime OCR: {index}/{len(records)} images', flush=True)
    summary = metrics([(row['reference'], row['prediction']) for row in predictions])
    summary['status_counts'] = dict(Counter(row['result'].get('status', 'unknown') for row in predictions))
    summary['review_required_count'] = sum(bool(row['result'].get('requires_review')) for row in predictions)
    return summary, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--tessdata-dir', type=Path)
    parser.add_argument('--psm-modes', type=parse_psm_modes,
                        help='Explicit runtime segmentation modes; otherwise use deployed settings')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve previous results')
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers and limit must be positive')
    if args.split:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        records = manifest['splits'][args.split]
        domain = manifest.get('domain', 'unspecified')
    else:
        records = load_benchmark(args.manifest)
        domain = 'user-labelled benchmark; provenance and label correctness require review'
    if not records:
        parser.error('Benchmark is empty')
    records = records[:args.limit] if args.limit else records
    with use_tessdata(args.tessdata_dir):
        source_hash = sha256(Path(ocr_service.__file__))
        summary, predictions = evaluate_records(records, args.workers, args.psm_modes)
        directory = os.environ.get('SINHALA_TESSDATA_DIR') or ocr_service._selection_config().get('tessdata_dir')
        report = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'domain': domain, 'split': args.split, 'subset_limit': args.limit,
                  'pipeline': 'app.services.ocr_service.extract_text; automatic language selection',
                  'ocr_service_sha256': source_hash,
                  'source_changed_during_evaluation': source_hash != sha256(Path(ocr_service.__file__)),
                  'manifest_sha256': sha256(args.manifest),
                  'sin_sha256': sha256(Path(directory) / 'sin.traineddata') if directory else None,
                  'runtime_psm_modes': list(args.psm_modes or ocr_service.runtime_psm_modes()),
                  'summary': summary, 'predictions': predictions,
                  'limitations': ['Synthetic results do not establish real screenshot accuracy.',
                                  'All samples, including empty/error outputs, are included in metrics.']}
    write_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()

"""Accuracy must use all 200 words and must never grade unverified references."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from word_benchmark import score_word_benchmark


class WordAccuracyChecks(unittest.TestCase):
    def fixture(self, folder, verified=True):
        refs = pd.DataFrame({'test_id': range(1, 201),
                             'path': [f'word_{i}.png' for i in range(200)],
                             'expected': ['জল'] * 200, 'reference_verified': [verified] * 200})
        refs.to_csv(folder / 'references.csv', index=False)
        pd.DataFrame({'test_id': range(1, 201), 'path': refs['path'],
                      'prediction': ['জল'] * 120 + ['মানুষ'] * 60 + [''] * 20,
                      'status': ['accepted'] * 110 + ['uncertain'] * 10 + ['accepted'] * 60 + ['error'] * 20,
                      'latency_ms': [10.0] * 200, 'error': [''] * 180 + ['decode failure'] * 20,
                      }).to_csv(folder / 'predictions.csv', index=False)

    def test_all_200_images_are_in_the_accuracy_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            results, metrics = score_word_benchmark(folder / 'references.csv', folder)
            self.assertEqual(len(results), 200)
            self.assertEqual(metrics['correct_predictions'], 120)
            self.assertEqual(metrics['exact_word_accuracy'], 0.60)
            self.assertEqual(metrics['accepted_word_accuracy_rejections_as_errors'], 0.55)
            self.assertEqual(metrics['inference_errors'], 20)

    def test_unreviewed_labels_do_not_produce_an_accuracy_percentage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder, verified=False)
            _, metrics = score_word_benchmark(folder / 'references.csv', folder)
            self.assertIsNone(metrics['exact_word_accuracy'])
            self.assertIsNone(metrics['correct_predictions'])
            self.assertEqual(metrics['verified_reference_images'], 0)

    def test_missing_reference_and_incomplete_inference_cannot_be_silently_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            refs = pd.read_csv(folder / 'references.csv')
            refs.loc[0, 'expected'] = ''
            refs.to_csv(folder / 'references.csv', index=False)
            _, metrics = score_word_benchmark(folder / 'references.csv', folder)
            self.assertIsNone(metrics['exact_word_accuracy'])
            predictions = pd.read_csv(folder / 'predictions.csv').iloc[:-1]
            predictions.to_csv(folder / 'predictions.csv', index=False)
            with self.assertRaises(ValueError):
                score_word_benchmark(folder / 'references.csv', folder)


if __name__ == '__main__':
    unittest.main()

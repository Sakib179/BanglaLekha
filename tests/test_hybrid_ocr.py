"""Inference checks. Model perturbations below are test doubles, never training."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/bangla-local-eda-matplotlib')

import h5py
import numpy as np
import torch
from PIL import Image

import hybrid_ocr as hybrid
from hybrid_evaluation import align_sequences, evaluate_words, vowel_rows


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'training_runs/macos_resnet18_efficientnetb0_v1/character/resnet18/best_model.pt'
IMAGES = ROOT / 'BanglaLekha-Isolated/Images'
HELPER = ROOT / 'models/vowel_helper'


def controlled_jol():
    pieces = []
    for class_id in [19, 39]:
        path = sorted((IMAGES / str(class_id)).glob('*.png'))[10]
        pixels = hybrid.base_ocr.foreground_gray(path)
        y, x = np.where(pixels > 32)
        image = Image.fromarray(pixels[y.min():y.max() + 1, x.min():x.max() + 1])
        image.thumbnail((100, 100))
        pieces.append(image)
    canvas = Image.new('L', (sum(p.width for p in pieces) + 50, 125), 0)
    left = 15
    for piece in pieces:
        canvas.paste(piece, (left, 10))
        left += piece.width + 20
    return canvas


class UtilityChecks(unittest.TestCase):
    def test_alignment_counts_insertions_and_deletions(self):
        _, distance = align_sequences('মানুষ', 'মনষ')
        self.assertEqual(distance, 2)
        self.assertEqual(align_sequences('জল', '')[1], 2)
        rows = vowel_rows('মানুষ', 'মনষ')
        self.assertEqual(sum(not r['sign_correct'] for r in rows), 2)

    def test_joining_and_prebase_sign_order(self):
        self.assertEqual(hybrid.join_tokens([{'base': 'ক', 'vowel': 'ি'}]), 'কি')
        self.assertEqual(hybrid.join_tokens([{'base': 'ম', 'vowel': 'া'},
                                            {'base': 'ন', 'vowel': 'ু'},
                                            {'base': 'ষ', 'vowel': ''}]), 'মানুষ')

    def test_blank_and_missing_resnet_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'ResNet'):
            hybrid.predict_word({'resnet': None}, Image.new('L', (30, 30)))
        with self.assertRaisesRegex(ValueError, 'Blank'):
            hybrid.base_ocr.prepare_word(Image.new('L', (30, 30)), hybrid.SETTINGS)

    def test_notebook_cells_compile_and_stay_short(self):
        notebook = json.loads((ROOT / '04_hybrid_word_recognition.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, '<hybrid-cell>', 'exec')
                self.assertLessEqual(len(source.splitlines()), 40)


@unittest.skipUnless(CHECKPOINT.is_file() and (HELPER / 'model.h5').is_file() and IMAGES.is_dir(),
                     'Download the helper and retain local model/images for integration checks.')
class FrozenModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.hash_before = hybrid.file_sha256(CHECKPOINT)
        cls.engine = hybrid.load_hybrid_engine(CHECKPOINT, helper_dir=HELPER)
        cls.image = controlled_jol()
        cls.result = hybrid.predict_word(cls.engine, cls.image)

    @classmethod
    def tearDownClass(cls):
        assert hybrid.file_sha256(CHECKPOINT) == cls.hash_before

    def test_original_helper_weights_load_bit_for_bit(self):
        # Compare every retained parameter and moving statistic to the original HDF5.
        with h5py.File(HELPER / 'model.h5', 'r') as handle:
            for layer in self.engine['vowel_head'].layers:
                arrays = [variable.value.detach().cpu().numpy() for variable in layer.weights]
                if not arrays:
                    continue
                group = handle['model_weights'][layer.name]
                names = group.attrs['weight_names']
                self.assertEqual(len(arrays), len(names))
                for current, name in zip(arrays, names):
                    name = name.decode() if isinstance(name, bytes) else name
                    np.testing.assert_array_equal(current, group[name][...])

    def test_both_models_are_frozen_and_only_vowel_head_is_exposed(self):
        engine = self.engine
        self.assertEqual(engine['vowel_head'].output_shape[-1], 11)
        self.assertEqual(len(engine['vowel_head'].trainable_variables), 0)
        with self.assertRaises(ValueError):
            engine['vowel_head'].get_layer('grapheme_root')
        self.assertFalse(engine['resnet']['model'].training)
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in engine['resnet']['model'].parameters()))

    def test_controlled_word_and_filename_do_not_supply_the_answer(self):
        self.assertEqual(self.result['word'], 'জল')
        self.assertTrue(all(t['base_source'] == 'trained_resnet18' for t in self.result['tokens']))
        self.assertEqual([t['base_class_id'] for t in self.result['tokens']], [19, 39])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'মানুষ.png'
            self.image.save(path)
            with patch('urllib.request.urlopen', side_effect=AssertionError('Inference must be offline')):
                self.assertEqual(hybrid.predict_word(self.engine, path)['word'], 'জল')

    def test_natural_manush_uses_our_three_base_classes(self):
        path = (ROOT / 'Handwritten word' /
                'PromitoLipi A versatile offline dataset of handwritten Bangla words and paragraphs' /
                'PromitoLipi/PromitoLipi2.1/Word Images(bmp)/1971.bmp')
        if not path.is_file():
            self.skipTest('Optional natural-word fixture is absent.')
        result = hybrid.predict_word(self.engine, path)
        self.assertEqual(result['word'], 'মানুষ')
        self.assertEqual([t['base_class_id'] for t in result['tokens']], [36, 31, 41])
        self.assertEqual([t['vowel'] for t in result['tokens']], ['া', 'ু', ''])

    def test_changing_resnet_changes_reconstruction_and_uniform_logits_reject(self):
        class ConstantModel(torch.nn.Module):
            def __init__(self, forced_class=None):
                super().__init__()
                self.forced_class = forced_class

            def forward(self, images):
                logits = torch.zeros((len(images), 84), device=images.device)
                if self.forced_class is not None:
                    logits[:, self.forced_class - 1] = 20
                return logits

        original = self.engine['resnet']['model']
        try:
            self.engine['resnet']['model'] = ConstantModel(39).eval()
            changed = hybrid.predict_word(self.engine, self.image)
            self.assertNotEqual(changed['word'], self.result['word'])
            self.assertTrue(all(t['base_class_id'] == 39 for t in changed['tokens']))
            self.engine['resnet']['model'] = ConstantModel().eval()
            uniform = hybrid.predict_word(self.engine, self.image)
            self.assertIsNone(uniform['accepted_word'])
            self.assertEqual(uniform['status'], 'uncertain')
        finally:
            self.engine['resnet']['model'] = original

    def test_csv_errors_and_rejections_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.image.save(folder / 'drawing.png')
            (folder / 'tests.csv').write_text('path,expected\ndrawing.png,জল\nmissing.png,মানুষ\n', encoding='utf-8')
            words, _, summary = evaluate_words(self.engine, folder / 'tests.csv', folder / 'reports')
            self.assertEqual(len(words), 2)
            self.assertEqual(summary['raw_word_accuracy'], 0.5)
            self.assertEqual(words.iloc[1]['status'], 'error')
            self.assertTrue(words.iloc[1]['error'])


if __name__ == '__main__':
    unittest.main()

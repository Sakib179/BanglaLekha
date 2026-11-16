"""Integration checks for inference only; these never train or update weights."""

import ast
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

import segmentation_ocr as ocr


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'training_runs/macos_resnet18_efficientnetb0_v1/character/resnet18/best_model.pt'
IMAGES = ROOT / 'BanglaLekha-Isolated/Images'


def character_sample(class_id):
    return sorted((IMAGES / str(class_id)).glob('*.png'))[10]


def controlled_jol():
    pieces = []
    for class_id in [19, 39]:
        pixels = ocr.foreground_gray(character_sample(class_id))
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


class PrototypeChecks(unittest.TestCase):
    def test_notebook_is_small_and_matches_module(self):
        notebook = json.loads((ROOT / '03_segmentation_prototype.ipynb').read_text())
        blocks = []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            compile(source, '<notebook-cell>', 'exec')
            self.assertLessEqual(len(source.splitlines()), 40)
            if 'prototype_source' in cell['metadata'].get('tags', []):
                blocks.append(source)

        original = ast.parse((ROOT / 'segmentation_ocr.py').read_text())
        original.body = original.body[1:]  # Ignore only the module docstring.
        embedded = ast.parse('\n'.join(blocks))
        self.assertEqual(ast.dump(original), ast.dump(embedded))

    def test_blank_inputs_are_rejected(self):
        for value in [0, 127, 255]:
            with self.assertRaisesRegex(ValueError, 'Blank'):
                ocr.prepare_word(Image.new('L', (100, 50), value), ocr.DEFAULT_SETTINGS)

    def test_polarity_and_transparency_preserve_ink(self):
        pixels = np.zeros((80, 100), dtype=np.uint8)
        pixels[20:60, 30:70] = 255
        np.testing.assert_array_equal(ocr.foreground_gray(pixels), ocr.foreground_gray(255 - pixels))

        rgba = Image.new('RGBA', (100, 80), (255, 255, 255, 0))
        rgba.paste((0, 0, 0, 255), (30, 20, 70, 60))
        np.testing.assert_array_equal(pixels, ocr.foreground_gray(rgba))

    def test_tiny_component_does_not_crash(self):
        mask = np.array([[255, 255]], dtype=np.uint8)
        self.assertEqual(ocr.split_wide_boxes(mask, [[0, 0, 2, 1]]), [[0, 0, 2, 1]])

    def test_unicode_reconstruction_only(self):
        # This tests joining text, NOT recognizing an image of মানুষ.
        tokens = [{'text': text, 'modifier': sign} for text, sign in
                  [('ম', 'া'), ('ন', 'ু'), ('ষ', '')]]
        self.assertEqual(ocr.assemble_word(tokens), 'মানুষ')
        self.assertEqual(ocr.assemble_word([{'text': 'অ', 'modifier': 'া'}]), 'আ')

    @unittest.skipUnless(IMAGES.is_dir(), 'Optional local character images are absent.')
    def test_preprocessing_matches_training_notebook_exactly(self):
        notebook = json.loads((ROOT / '02-model-training.ipynb').read_text())
        names = {'normalise_ink', 'image_canvas', 'canvas_tensor'}
        definitions = []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if not any(f'def {name}(' in source for name in names):
                continue
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    definitions.append(node)

        self.assertEqual(len(definitions), 3)
        namespace = dict(np=np, Image=Image, ImageOps=ImageOps, TF=TF, random=random)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<training-preprocess>', 'exec'), namespace)

        for class_id in [1, 19, 39, 61, 84]:
            with Image.open(character_sample(class_id)) as image:
                original = namespace['canvas_tensor'](namespace['image_canvas'](image, 128, 128))
                current = ocr.character_tensor(ocr.foreground_gray(image))
            torch.testing.assert_close(original, current, rtol=0, atol=0)

    @unittest.skipUnless(CHECKPOINT.is_file() and IMAGES.is_dir(), 'Saved model and local images required.')
    def test_real_checkpoint_joining_and_filename_independence(self):
        torch.set_num_threads(2)
        engine = ocr.load_engine(CHECKPOINT, ROOT / 'labels.json')
        image = controlled_jol()
        before = {name: value.clone() for name, value in engine['model'].named_buffers()}
        first = ocr.predict_word(engine, image)

        self.assertEqual(first['word'], 'জল')
        self.assertFalse(engine['model'].training)
        self.assertFalse(first['confidence_is_calibrated'])
        self.assertTrue(all(p.grad is None for p in engine['model'].parameters()))
        for name, value in engine['model'].named_buffers():
            torch.testing.assert_close(before[name], value, rtol=0, atol=0)

        with tempfile.TemporaryDirectory() as folder:
            # Filenames cannot supply or force an expected answer.
            path = Path(folder) / 'মানুষ.png'
            image.save(path)
            self.assertEqual(ocr.predict_word(engine, path)['word'], first['word'])


if __name__ == '__main__':
    unittest.main()

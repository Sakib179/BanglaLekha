"""Validate images and resolve the existing local model assets."""
import base64
import binascii
import os
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def model_paths():
    deployed = ROOT / 'models/resnet18.pt'
    original = ROOT / 'training_runs/macos_resnet18_efficientnetb0_v1/character/resnet18/best_model.pt'
    checkpoint = Path(os.environ.get('MODEL_PATH', str(deployed if deployed.is_file() else original)))
    helper = Path(os.environ.get('VOWEL_HELPER_DIR', str(ROOT / 'models/vowel_helper')))
    return checkpoint, helper


def read_image(content):
    """Decode bytes; put transparent handwriting on white rather than black."""
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError('Choose an image smaller than 10 MB.')
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.width * opened.height > 12_000_000:
                raise ValueError('Choose an image with fewer than 12 million pixels.')
            image = ImageOps.exif_transpose(opened).convert('RGBA')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('This image could not be opened. Try a PNG or JPG file.') from error
    page = Image.new('RGBA', image.size, 'white')
    return Image.alpha_composite(page, image).convert('RGB')


def canvas_image(data_url):
    if not isinstance(data_url, str) or not data_url.startswith('data:image/png;base64,'):
        raise ValueError('The drawing could not be read. Clear the canvas and try again.')
    if len(data_url) > MAX_IMAGE_BYTES * 4 // 3 + 100:
        raise ValueError('The drawing is too large.')
    try:
        content = base64.b64decode(data_url.split(',', 1)[1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('The drawing data is incomplete. Please try again.') from error
    return read_image(content)

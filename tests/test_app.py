"""Boundary checks for image upload and drawing payloads; no training required."""
import base64
from io import BytesIO
import unittest
from PIL import Image
from ocr_service import canvas_image, read_image


class InputChecks(unittest.TestCase):
    def test_transparent_canvas_uses_white_paper(self):
        image = Image.new('RGBA', (8, 8), (0, 0, 0, 0))
        image.putpixel((4, 4), (0, 0, 0, 255))
        content = BytesIO()
        image.save(content, format='PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(content.getvalue()).decode()
        decoded = canvas_image(uri)
        self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(decoded.getpixel((4, 4)), (0, 0, 0))

    def test_invalid_or_oversized_upload_is_rejected(self):
        for data in [b'', b'not an image', b'x' * (10 * 1024 * 1024 + 1)]:
            with self.assertRaises(ValueError):
                read_image(data)

    def test_invalid_canvas_payload_is_rejected(self):
        for payload in [None, 'https://example.com/image.png', 'data:image/png;base64,???']:
            with self.assertRaises(ValueError):
                canvas_image(payload)


if __name__ == '__main__':
    unittest.main()

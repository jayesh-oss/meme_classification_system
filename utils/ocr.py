import os

import pytesseract
from PIL import Image

_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def extract_text(image_path):
    try:
        with Image.open(image_path) as image:
            return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""

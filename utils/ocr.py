import os
import shutil

import pytesseract
from PIL import Image

_tesseract_cmd = os.getenv("TESSERACT_CMD")
if not _tesseract_cmd and shutil.which("tesseract") is None:
    _default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_tesseract):
        _tesseract_cmd = _default_tesseract
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def extract_text(image_path):
    try:
        with Image.open(image_path) as image:
            return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""

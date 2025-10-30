import pytesseract
from PIL import Image
import torch
from transformers import pipeline

# Point pytesseract to installed tesseract executable (Windows users need this)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Use Hugging Face pipeline for text classification
device = 0 if torch.cuda.is_available() else -1
text_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=device
)

# Map toxic-bert labels to our categories
CATEGORY_MAP = {
    "toxic": "Offensive",
    "severe_toxic": "Dark",
    "obscene": "Offensive",
    "threat": "Dark",
    "insult": "Offensive",
    "identity_hate": "Political",
    "neutral": "Wholesome",
}

def analyze_text(filepath):
    """
    Extract text from meme using Tesseract OCR,
    then classify it using Hugging Face toxic-bert.
    """

    # Extract text from image
    text = pytesseract.image_to_string(Image.open(filepath))
    text = text.strip()

    if not text:
        return "Wholesome"  # fallback if no text is detected

    # Run through model
    results = text_classifier(text, top_k=None)

    # Pick the label with highest score
    top_result = max(results, key=lambda x: x["score"])
    label = top_result["label"].lower()

    # Map to our categories
    return CATEGORY_MAP.get(label, "Wholesome")

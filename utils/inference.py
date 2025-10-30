import pytesseract
from PIL import Image
from transformers import pipeline

# load zero-shot classification model once
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# candidate categories
categories = ["wholesome", "dark", "political", "promotional", "offensive"]

def classify_meme(image_path):
    # Extract text from meme
    text = pytesseract.image_to_string(Image.open(image_path))

    if not text.strip():
        text = "This meme has no readable text"

    # Classify with zero-shot
    result = classifier(text, candidate_labels=categories)
    predicted = result["labels"][0]  # highest confidence

    return predicted

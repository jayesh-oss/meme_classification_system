import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# Load pre-trained ViT (Vision Transformer)
device = 0 if torch.cuda.is_available() else -1
model_name = "google/vit-base-patch16-224"

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name).to("cuda" if device == 0 else "cpu")

# Custom category mapping (adjust as you like)
CATEGORY_MAP = {
    "dog": "Wholesome",
    "cat": "Wholesome",
    "person": "Political",
    "flag": "Political",
    "logo": "Promotional",
    "weapon": "Dark",
    "skull": "Dark",
    "default": "Wholesome"
}

def analyze_image(filepath):
    """
    Classify meme image when no text is available.
    Uses Vision Transformer to get labels and maps them to categories.
    """
    image = Image.open(filepath).convert("RGB")

    # Preprocess
    inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Get label from pretrained model
    label = model.config.id2label[predicted_class_idx].lower()

    # Map to your categories
    return CATEGORY_MAP.get(label, CATEGORY_MAP["default"])

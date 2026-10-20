import os

from utils.ocr import extract_text

CATEGORIES = ["Wholesome", "Dark", "Political", "Promotional", "Offensive"]
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_FUSION_HEAD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "meme_head.pt",
)
HYPOTHESIS_TEMPLATE = "This is a {} meme."

_zero_shot = None
_fusion = None


def _device():
    import torch

    return 0 if torch.cuda.is_available() else -1


def _get_zero_shot():
    global _zero_shot

    if _zero_shot is None:
        from transformers import pipeline

        device = _device()
        _zero_shot = pipeline(
            "zero-shot-image-classification",
            model=os.getenv("CLIP_MODEL", DEFAULT_CLIP_MODEL),
            device=device,
        )
        print(f"CLIP zero-shot device: {'cuda' if device == 0 else 'cpu'}")
    return _zero_shot


def _fusion_head_path():
    return os.getenv("FUSION_HEAD", DEFAULT_FUSION_HEAD)


def _get_fusion():
    global _fusion

    if _fusion is None:
        from utils.fusion import FusionClassifier

        _fusion = FusionClassifier(_fusion_head_path())
    return _fusion


def classify_meme(image_path):
    text = extract_text(image_path)
    has_text = bool(text)

    if os.path.exists(_fusion_head_path()):
        try:
            probabilities = _get_fusion().predict(image_path, text)
            scores = [
                {"label": label, "score": round(probability * 100, 1)}
                for label, probability in probabilities
            ]
            scores.sort(key=lambda item: item["score"], reverse=True)
            return {
                "prediction": scores[0]["label"],
                "scores": scores,
                "extracted_text": text if has_text else "No text detected.",
                "has_text": has_text,
            }
        except Exception as error:
            print(f"Fusion head unavailable, falling back to zero-shot: {error}")

    results = _get_zero_shot()(
        image_path,
        candidate_labels=CATEGORIES,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    scores = [
        {"label": item["label"], "score": round(float(item["score"]) * 100, 1)}
        for item in results
    ]
    return {
        "prediction": scores[0]["label"],
        "scores": scores,
        "extracted_text": text if has_text else "No text detected.",
        "has_text": has_text,
    }

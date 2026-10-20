import argparse
import json
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.fusion import FusionClassifier
from utils.inference import classify_zero_shot
from utils.ocr import extract_text


def canonical_label(label, class_names):
    lookup = {name.lower(): name for name in class_names}
    return lookup.get(label.lower(), label)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP zero-shot or the trained fusion head on a labeled folder."
    )
    parser.add_argument("--data-dir", default="data/test")
    parser.add_argument("--mode", choices=["auto", "zero-shot", "fusion"], default="auto")
    parser.add_argument("--head", default="models/meme_head.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = ImageFolder(args.data_dir)
    samples = dataset.samples[: args.limit] if args.limit else dataset.samples
    y_true = [dataset.classes[label] for _, label in samples]

    mode = args.mode
    if mode == "auto":
        mode = "fusion" if os.path.exists(args.head) else "zero-shot"

    if mode == "zero-shot":
        y_pred = []
        for path, _ in tqdm(samples, desc="zero-shot"):
            result = classify_zero_shot(path)
            y_pred.append(canonical_label(result["prediction"], dataset.classes))
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        classifier = FusionClassifier(args.head, device=device)
        y_pred = []
        for path, _ in tqdm(samples, desc="fusion"):
            probabilities = classifier.predict(path, extract_text(path))
            label = max(probabilities, key=lambda item: item[1])[0]
            y_pred.append(canonical_label(label, dataset.classes))

    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )

    print(f"Mode: {mode}")
    print(f"Images: {len(samples)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(f"Labels: {labels}")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = {
            "mode": mode,
            "images": len(samples),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "labels": labels,
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist(),
        }
        with open(args.output, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()

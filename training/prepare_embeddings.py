import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.fusion import extract_image_features
from utils.ocr import extract_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP image and OCR text embeddings for meme training."
    )
    parser.add_argument("--data-dir", default="data/memes")
    parser.add_argument("--output", default="data/embeddings.npz")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--text-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    dataset = ImageFolder(args.data_dir)
    clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    text_model = SentenceTransformer(args.text_model, device=str(device))

    image_embeddings = []
    text_embeddings = []
    labels = []

    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), args.batch_size)):
            batch = [
                dataset[index]
                for index in range(
                    start, min(start + args.batch_size, len(dataset))
                )
            ]
            images = [image.convert("RGB") for image, _ in batch]
            batch_labels = [label for _, label in batch]
            batch_texts = [
                extract_text(dataset.samples[start + offset][0]) or "no text"
                for offset in range(len(batch))
            ]

            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = extract_image_features(clip, inputs)
            features = features / features.norm(dim=-1, keepdim=True)

            image_embeddings.append(features.cpu().numpy())
            text_embeddings.append(
                text_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )
            labels.extend(batch_labels)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        args.output,
        image=np.concatenate(image_embeddings),
        text=np.concatenate(text_embeddings),
        labels=np.asarray(labels),
        class_names=np.asarray(dataset.classes),
        clip_model=args.clip_model,
        text_model=args.text_model,
    )
    print(f"Saved {len(labels)} embeddings to {args.output}")
    print(f"Classes: {dataset.classes}")


if __name__ == "__main__":
    main()

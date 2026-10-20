import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.fusion import FusionHead


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CLIP + OCR text fusion head."
    )
    parser.add_argument("--embeddings", default="data/embeddings.npz")
    parser.add_argument("--output", default="models/meme_head.pt")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    data = np.load(args.embeddings)
    image = torch.tensor(data["image"], dtype=torch.float32)
    text = torch.tensor(data["text"], dtype=torch.float32)
    labels_np = data["labels"].astype(np.int64)
    class_names = data["class_names"].tolist()
    clip_model = str(data["clip_model"])
    text_model = str(data["text_model"])

    indices = np.arange(len(labels_np))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        stratify=labels_np,
        random_state=args.seed,
    )

    classes = np.unique(labels_np[train_idx])
    weights = compute_class_weight("balanced", classes=classes, y=labels_np[train_idx])
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = FusionHead(
        image_dim=image.shape[1],
        text_dim=text.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=len(class_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_image = image[train_idx].to(device)
    train_text = text[train_idx].to(device)
    train_labels = torch.tensor(labels_np[train_idx], dtype=torch.long).to(device)
    val_image = image[val_idx].to(device)
    val_text = text[val_idx].to(device)

    best_f1 = -1.0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_image, train_text)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_image, val_text)
            predictions = val_logits.argmax(dim=-1).cpu().numpy()
        macro_f1 = f1_score(labels_np[val_idx], predictions, average="macro")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d} loss {loss.item():.4f} "
                f"val_macro_f1 {macro_f1:.4f}"
            )

        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(val_image, val_text)
        predictions = val_logits.argmax(dim=-1).cpu().numpy()

    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(classification_report(labels_np[val_idx], predictions, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(labels_np[val_idx], predictions))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": class_names,
            "head_config": {
                "image_dim": image.shape[1],
                "text_dim": text.shape[1],
                "hidden_dim": args.hidden_dim,
                "num_classes": len(class_names),
            },
            "clip_model": clip_model,
            "text_model": text_model,
        },
        args.output,
    )
    print(f"Saved fusion head to {args.output}")


if __name__ == "__main__":
    main()

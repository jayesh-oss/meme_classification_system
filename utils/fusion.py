import os

import torch
import torch.nn as nn


def extract_image_features(clip, inputs):
    outputs = clip.get_image_features(**inputs)
    return getattr(outputs, "pooler_output", outputs)


class FusionHead(nn.Module):
    def __init__(self, image_dim=512, text_dim=384, hidden_dim=256, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_features, text_features):
        return self.classifier(torch.cat([image_features, text_features], dim=-1))


class FusionClassifier:
    def __init__(self, head_path, device=None):
        from sentence_transformers import SentenceTransformer
        from transformers import CLIPModel, CLIPProcessor

        if not os.path.exists(head_path):
            raise FileNotFoundError(head_path)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Fusion head device: {self.device}")
        checkpoint = torch.load(head_path, map_location="cpu")

        self.labels = checkpoint["labels"]
        self.clip_model_name = checkpoint.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self.text_model_name = checkpoint.get(
            "text_model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.clip = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.text_model = SentenceTransformer(
            self.text_model_name, device=str(self.device)
        )
        self.head = FusionHead(**checkpoint["head_config"]).to(self.device).eval()
        self.head.load_state_dict(checkpoint["state_dict"])

    @torch.no_grad()
    def predict(self, image_path, text):
        from PIL import Image

        with Image.open(image_path) as image:
            inputs = self.processor(
                images=image.convert("RGB"), return_tensors="pt"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        image_features = extract_image_features(self.clip, inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_model.encode(
            [text or "no text"],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(self.device)

        logits = self.head(image_features, text_features)
        probabilities = logits.softmax(dim=-1)[0].cpu().tolist()
        return list(zip(self.labels, probabilities))

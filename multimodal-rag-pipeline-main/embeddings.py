"""
Text embeddings via sentence-transformers.
Image (and cross-modal text) embeddings via open_clip.

Using CLIP for images lets us embed a plain-text query into the SAME
vector space as the images, so a text question can retrieve relevant
diagrams/screenshots directly.
"""
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import open_clip

import config

_device = "cuda" if torch.cuda.is_available() else "cpu"


class TextEmbedder:
    def __init__(self, model_name=config.TEXT_EMBED_MODEL):
        self.model = SentenceTransformer(model_name, device=_device)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class ClipEmbedder:
    """Embeds images, and embeds text queries into the same space
    as those images for cross-modal retrieval."""

    def __init__(self, model_name=config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(_device).eval()

    @torch.no_grad()
    def embed_images(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = [self.preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in image_paths]
        batch = torch.cat(images).to(_device)
        features = self.model.encode_image(batch)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    @torch.no_grad()
    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(texts).to(_device)
        features = self.model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

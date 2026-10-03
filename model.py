"""Pretrained frame-level deepfake classifier.

The checkpoint is downloaded from Hugging Face on first use and cached locally
by Transformers. Video-level predictions are produced by aggregating sampled
frame probabilities.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "hamzenium/ViT-Deepfake-Classifier"
FAKE_LABELS = {"fake", "deepfake", "1"}

@lru_cache(maxsize=1)
def load_model():
    """Load and cache the pretrained processor and model."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

def fake_probability(image: Image.Image) -> float:
    """Return the model probability that a frame is fake."""
    processor, model = load_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    fake_index = None
    for index, label in model.config.id2label.items():
        normalized = str(label).strip().lower()
        if normalized in FAKE_LABELS or "fake" in normalized:
            fake_index = int(index)
            break
    if fake_index is None:
        fake_index = 1
    return float(probabilities[fake_index].item())
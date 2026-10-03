"""Temporal VideoMAE deepfake classifier."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

MODEL_ID = "SoraExplora/VideoMae"
NUM_FRAMES = 16

@lru_cache(maxsize=1)
def load_model():
    """Load the fine-tuned temporal video classifier and processor once."""
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

def predict_clip(frames: list[Image.Image]) -> tuple[float, float]:
    """Classify one ordered 16-frame clip as real/fake."""
    if len(frames) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, received {len(frames)}.")

    processor, model = load_model()
    inputs = processor(frames, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    # The model card defines class 0 as real and class 1 as fake.
    return float(probabilities[0]), float(probabilities[1])

def sample_frame_indices(total_frames: int, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Select an ordered temporal sequence spanning the video."""
    if total_frames <= 0:
        raise ValueError("Video contains no readable frames.")
    return np.linspace(0, total_frames - 1, num_frames).astype(int)
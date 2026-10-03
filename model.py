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
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def predict_clip(frames: list[Image.Image]) -> tuple[float, float]:
    if len(frames) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, received {len(frames)}.")
    processor, model = load_model()
    inputs = processor(frames, return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return float(probabilities[0]), float(probabilities[1])


def sample_clip_starts(
    total_frames: int,
    clip_frames: int = NUM_FRAMES,
    max_clips: int = 5,
) -> list[tuple[int, int]]:
    """Return up to max_clips evenly distributed temporal windows."""
    if total_frames < clip_frames:
        raise ValueError(f"Video needs at least {clip_frames} frames.")
    clip_count = min(max_clips, max(1, total_frames // clip_frames))
    max_start = total_frames - clip_frames
    starts = np.linspace(0, max_start, clip_count).astype(int)
    return [(int(start), int(start + clip_frames - 1)) for start in starts]

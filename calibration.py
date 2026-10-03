"""Temperature scaling utilities for VideoMAE probability calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


def load_temperature(path: Path = CALIBRATION_PATH) -> float:
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. Run calibrate.py on a labeled "
            "validation set before using calibrated confidence."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    temperature = float(data["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be a finite value greater than zero.")
    return temperature


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def calibrate_logit(raw_fake_logit: float, temperature: float) -> float:
    """Apply fitted temperature scaling to a fake-class logit."""
    return sigmoid(raw_fake_logit / temperature)

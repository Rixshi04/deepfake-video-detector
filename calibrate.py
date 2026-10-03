"""Fit a temperature-scaling calibrator from labeled validation predictions.

CSV format:
    raw_fake_logit,label
    1.42,1
    -0.83,0

The logit must be the video's raw fake-vs-real logit after aggregating its
temporal clips in the same way as the application.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def nll(logits: list[float], labels: list[int], temperature: float) -> float:
    total = 0.0
    for logit, label in zip(logits, labels):
        p = min(max(sigmoid(logit / temperature), 1e-7), 1.0 - 1e-7)
        total += -(label * math.log(p) + (1 - label) * math.log(1 - p))
    return total / len(labels)


def fit_temperature(logits: list[float], labels: list[int]) -> float:
    if len(logits) < 20:
        raise ValueError("Use at least 20 labeled validation videos for calibration.")
    if len(set(labels)) != 2:
        raise ValueError("Validation data must contain both real (0) and fake (1) labels.")

    # Deterministic coarse-to-fine search over log-temperature.
    lo, hi = math.log(0.05), math.log(10.0)
    for _ in range(8):
        grid = [lo + (hi - lo) * i / 40 for i in range(41)]
        best = min(grid, key=lambda x: nll(logits, labels, math.exp(x)))
        center = best
        step = (hi - lo) / 40
        lo, hi = center - step, center + step
    return math.exp(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit VideoMAE temperature scaling.")
    parser.add_argument("csv", type=Path, help="CSV containing raw_fake_logit,label.")
    parser.add_argument("--output", type=Path, default=Path("calibration.json"))
    args = parser.parse_args()

    logits: list[float] = []
    labels: list[int] = []
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            logits.append(float(row["raw_fake_logit"]))
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError("Labels must be 0 for real or 1 for fake.")
            labels.append(label)

    temperature = fit_temperature(logits, labels)
    before = nll(logits, labels, 1.0)
    after = nll(logits, labels, temperature)

    payload = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "validation_samples": len(labels),
        "nll_before": before,
        "nll_after": after,
        "label_definition": {"0": "real", "1": "fake"},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved calibration to {args.output}")
    print(f"Temperature: {temperature:.6f}")
    print(f"Validation NLL: {before:.6f} -> {after:.6f}")


if __name__ == "__main__":
    main()

"""Build validation predictions from labeled videos and fit calibration.

Expected dataset layout:
    validation_videos/
      real/
        video1.mp4
        video2.mp4
      fake/
        video3.mp4
        video4.mp4

Each video produces one raw fake-vs-real logit by averaging the logits from
the same evenly distributed 16-frame temporal clips used by the app.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
from PIL import Image

from calibrate import fit_temperature, nll
from model import NUM_FRAMES, load_model, predict_clip, sample_clip_starts

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_CLIPS = 5


def read_clip(video_path: Path, start_frame: int, end_frame: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the video.")
    try:
        indices = [
            int(start_frame + round(i * (end_frame - start_frame) / max(NUM_FRAMES - 1, 1)))
            for i in range(NUM_FRAMES)
        ]
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {index}.")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        capture.release()


def video_logit(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the video.")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    clips = sample_clip_starts(total_frames, NUM_FRAMES, MAX_CLIPS)
    logits = []
    for start, end in clips:
        real_score, fake_score = predict_clip(read_clip(video_path, start, end))
        logits.append(math.log(max(fake_score, 1e-7) / max(real_score, 1e-7)))
    return sum(logits) / len(logits)


def collect_videos(root: Path) -> list[tuple[Path, int]]:
    if not root.exists():
        raise FileNotFoundError(f"Validation directory not found: {root}")

    items = []
    for folder_name, label in (("real", 0), ("fake", 1)):
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected labeled folder: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                items.append((path, label))

    if not items:
        raise ValueError("No supported validation videos were found.")
    return items


def build_predictions(root: Path, output: Path) -> tuple[list[float], list[int]]:
    items = collect_videos(root)
    load_model()
    logits, labels = [], []

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "label", "raw_fake_logit"],
        )
        writer.writeheader()

        for index, (video, label) in enumerate(items, 1):
            print(f"[{index}/{len(items)}] {video}")
            try:
                logit = video_logit(video)
            except Exception as exc:
                raise RuntimeError(f"Failed on {video}: {exc}") from exc
            writer.writerow(
                {
                    "video": str(video.relative_to(root)),
                    "label": label,
                    "raw_fake_logit": f"{logit:.8f}",
                }
            )
            handle.flush()
            logits.append(logit)
            labels.append(label)

    return logits, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate validation_predictions.csv and calibration.json from labeled videos."
    )
    parser.add_argument(
        "validation_dir",
        type=Path,
        help="Folder containing real/ and fake/ subfolders.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("validation_predictions.csv"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
    )
    args = parser.parse_args()

    logits, labels = build_predictions(args.validation_dir, args.predictions)
    if len(logits) < 20:
        raise ValueError(
            f"Found {len(logits)} validation videos. At least 20 are required for calibration."
        )
    if len(set(labels)) != 2:
        raise ValueError("Validation data must contain both real and fake videos.")

    temperature = fit_temperature(logits, labels)
    before = nll(logits, labels, 1.0)
    after = nll(logits, labels, temperature)

    import json
    args.calibration.write_text(
        json.dumps(
            {
                "method": "temperature_scaling",
                "temperature": temperature,
                "validation_samples": len(labels),
                "nll_before": before,
                "nll_after": after,
                "label_definition": {"0": "real", "1": "fake"},
                "max_clips": MAX_CLIPS,
                "frames_per_clip": NUM_FRAMES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved predictions to {args.predictions}")
    print(f"Saved calibration to {args.calibration}")
    print(f"Temperature: {temperature:.6f}")
    print(f"Validation NLL: {before:.6f} -> {after:.6f}")


if __name__ == "__main__":
    main()

"""Gradio application for calibrated multi-clip temporal deepfake detection."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import gradio as gr
from PIL import Image

from calibration import calibrate_logit, load_temperature
from model import NUM_FRAMES, load_model, predict_clip, sample_clip_starts

MAX_CLIPS = 5


def read_clip(video_path: str, start_frame: int, end_frame: int):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video.")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        actual_indices = [
            int(start_frame + round(i * (end_frame - start_frame) / max(NUM_FRAMES - 1, 1)))
            for i in range(NUM_FRAMES)
        ]
        frames = []
        for index in actual_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {index}.")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames, {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration,
        }
    finally:
        capture.release()


def analyze_video(video_path: str | None) -> tuple[str, str]:
    if not video_path:
        return "Upload a video to begin.", ""
    path = Path(video_path)
    if not path.exists():
        return "The uploaded video could not be found.", ""

    try:
        temperature = load_temperature()
        load_model()

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return "OpenCV could not open this video.", ""
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        clips = sample_clip_starts(total_frames, NUM_FRAMES, MAX_CLIPS)
        raw_logits = []
        calibrated_scores = []
        metadata = None

        for start, end in clips:
            frames, metadata = read_clip(str(path), start, end)
            real_score, fake_score = predict_clip(frames)
            raw_logit = math.log(
                max(fake_score, 1e-7) / max(real_score, 1e-7)
            )
            raw_logits.append(raw_logit)
            calibrated_scores.append(calibrate_logit(raw_logit, temperature))

        aggregate_logit = sum(raw_logits) / len(raw_logits)
        calibrated_fake = calibrate_logit(aggregate_logit, temperature)
        calibrated_real = 1.0 - calibrated_fake
        verdict = "LIKELY DEEPFAKE" if calibrated_fake >= 0.5 else "LIKELY REAL"
        predicted_class_confidence = max(calibrated_fake, calibrated_real)

        lines = ["### Per-clip calibrated predictions"]
        for number, score in enumerate(calibrated_scores, 1):
            label = "DEEPFAKE" if score >= 0.5 else "REAL"
            lines.append(
                f"- **Clip {number}:** {label} — calibrated fake probability **{score:.1%}**"
            )

        score_range = max(calibrated_scores) - min(calibrated_scores)
        report = "\n".join(lines) + (
            "\n\n### Calibrated overall result\n"
            f"- **Verdict:** **{verdict}**\n"
            f"- **Calibrated fake probability:** **{calibrated_fake:.1%}**\n"
            f"- **Calibrated real probability:** **{calibrated_real:.1%}**\n"
            f"- **Clip score range:** **{min(calibrated_scores):.1%} – {max(calibrated_scores):.1%}**\n"
            f"- **Clip score variation:** **{score_range:.1%}**\n"
            f"- **Temporal clips:** **{len(calibrated_scores)} × {NUM_FRAMES} frames**\n"
            f"- **Predicted-class confidence:** **{predicted_class_confidence:.1%}**"
        )

        if metadata:
            report += (
                f"\n- **Resolution:** {metadata['width']} × {metadata['height']}"
                f"\n- **FPS:** {metadata['fps']:.2f}"
                f"\n- **Duration:** {metadata['duration']:.2f} seconds"
            )

        note = (
            f"Calibration method: temperature scaling, fitted on labeled validation "
            f"videos. Fitted temperature = **{temperature:.4f}**. Per-clip probabilities "
            "are calibrated with the same temperature. The overall probability is computed "
            "from the mean raw fake-vs-real logit across temporal clips, then temperature-scaled. "
            "Calibration is only meaningful when the calibration set matches the target data distribution."
        )
        return report, note
    except Exception as exc:
        return (
            "Calibrated inference is unavailable.",
            f"Technical detail: {exc}",
        )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Temporal Deepfake Video Detector") as demo:
        gr.Markdown(
            "# Temporal Deepfake Video Detector\n"
            "VideoMAE analyzes multiple ordered clips with calibrated probabilities."
        )
        video = gr.Video(label="Upload a video", type="filepath")
        analyze = gr.Button("Analyze Video", variant="primary")
        result = gr.Markdown()
        note = gr.Markdown()
        analyze.click(fn=analyze_video, inputs=video, outputs=[result, note])
    return demo


if __name__ == "__main__":
    build_app().launch()

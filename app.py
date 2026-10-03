"""Gradio application for multi-clip temporal deepfake detection."""
from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
from PIL import Image

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
        indices = sample_clip_starts(end_frame - start_frame + 1, NUM_FRAMES)
        indices = [start_frame + i[0] for i in indices[:1]]
        if not indices:
            indices = [start_frame]
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
        load_model()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return "OpenCV could not open this video.", ""
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        clips = sample_clip_starts(total_frames, NUM_FRAMES, MAX_CLIPS)
        scores = []
        metadata = None

        for start, end in clips:
            frames, metadata = read_clip(str(path), start, end)
            _, fake_score = predict_clip(frames)
            scores.append(fake_score)

        mean_score = sum(scores) / len(scores)
        score_range = max(scores) - min(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5

        consistency = max(0.0, 1.0 - min(std_dev / 0.5, 1.0))
        decision_strength = abs(mean_score - 0.5) * 2
        confidence = 0.5 * decision_strength + 0.5 * consistency

        verdict = "LIKELY DEEPFAKE" if mean_score >= 0.5 else "LIKELY REAL"

        lines = ["### Per-clip predictions"]
        for number, score in enumerate(scores, 1):
            label = "DEEPFAKE" if score >= 0.5 else "REAL"
            lines.append(
                f"- **Clip {number}:** {label} — fake probability **{score:.1%}**"
            )

        report = "\n".join(lines) + (
            "\n\n### Overall result\n"
            f"- **Verdict:** **{verdict}**\n"
            f"- **Mean fake probability:** **{mean_score:.1%}**\n"
            f"- **Score range:** **{min(scores):.1%} – {max(scores):.1%}**\n"
            f"- **Score variation:** **{score_range:.1%}**\n"
            f"- **Score standard deviation:** **{std_dev:.1%}**\n"
            f"- **Temporal clips:** **{len(scores)} × {NUM_FRAMES} frames**\n"
            f"- **Overall confidence summary:** **{confidence:.1%}**"
        )

        if metadata:
            report += (
                f"\n- **Resolution:** {metadata['width']} × {metadata['height']}"
                f"\n- **FPS:** {metadata['fps']:.2f}"
                f"\n- **Duration:** {metadata['duration']:.2f} seconds"
            )

        note = (
            "The confidence summary combines decision strength (distance from 50%) "
            "with consistency across clips. It is a presentation metric, not a "
            "calibrated probability or forensic certainty."
        )
        return report, note
    except Exception as exc:
        return "Temporal multi-clip inference failed.", f"Technical detail: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Temporal Deepfake Video Detector") as demo:
        gr.Markdown(
            "# Temporal Deepfake Video Detector\n"
            "VideoMAE analyzes multiple ordered clips to expose temporal score variation."
        )
        video = gr.Video(label="Upload a video", type="filepath")
        analyze = gr.Button("Analyze Video", variant="primary")
        result = gr.Markdown()
        note = gr.Markdown()
        analyze.click(fn=analyze_video, inputs=video, outputs=[result, note])
    return demo


if __name__ == "__main__":
    build_app().launch()

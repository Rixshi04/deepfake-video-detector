"""Deepfake video analysis demo.

This project is intentionally a pipeline/UI prototype. It does not claim to
classify videos as real or fake without a trained model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import gradio as gr


def analyze_video(video_path: str | None) -> tuple[str, str]:
    """Analyze basic video properties and frame statistics."""
    if not video_path:
        return "Upload a video to begin.", ""

    path = Path(video_path)
    if not path.exists():
        return "The uploaded video could not be found.", ""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return "OpenCV could not open this video.", ""

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0

        sample_count = min(30, frame_count) if frame_count > 0 else 0
        brightness_values: list[float] = []

        if sample_count:
            step = max(frame_count // sample_count, 1)
            frame_index = 0

            while frame_index < frame_count and len(brightness_values) < sample_count:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(float(gray.mean()))
                frame_index += step

        average_brightness = (
            sum(brightness_values) / len(brightness_values)
            if brightness_values
            else 0.0
        )

        report = (
            "### Video analysis\n"
            f"- **Resolution:** {width} × {height}\n"
            f"- **Frames:** {frame_count:,}\n"
            f"- **FPS:** {fps:.2f}\n"
            f"- **Duration:** {duration:.2f} seconds\n"
            f"- **Frames sampled:** {len(brightness_values)}\n"
            f"- **Average brightness:** {average_brightness:.2f}/255\n\n"
            "**Detection status:** Demo analysis only — no trained deepfake "
            "classifier is being used."
        )

        return report, (
            "This result describes video properties and sampled-frame statistics. "
            "It must not be interpreted as a real/fake prediction."
        )
    finally:
        capture.release()


def build_app() -> gr.Blocks:
    """Build the Gradio application."""
    with gr.Blocks(title="Deepfake Video Analysis Demo") as demo:
        gr.Markdown(
            "# Deepfake Video Analysis Demo\n"
            "A safe OpenCV + Gradio prototype for video-frame analysis.\n\n"
            "> **Note:** This is not a trained deepfake detector."
        )

        video = gr.Video(label="Upload a video", type="filepath")
        analyze = gr.Button("Analyze Video", variant="primary")
        result = gr.Markdown()
        note = gr.Markdown()

        analyze.click(
            fn=analyze_video,
            inputs=video,
            outputs=[result, note],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()

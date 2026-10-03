"""Deepfake video analysis application."""

from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
from PIL import Image

from model import fake_probability, load_model

MAX_SAMPLES = 24

def analyze_video(video_path: str | None) -> tuple[str, str]:
    """Run frame-level deepfake inference and aggregate it to a video score."""
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
        if frame_count <= 0:
            return "The video contains no readable frames.", ""
        sample_count = min(MAX_SAMPLES, frame_count)
        step = max(frame_count // sample_count, 1)
        scores: list[float] = []
        load_model()
        for frame_index in range(0, frame_count, step):
            if len(scores) >= MAX_SAMPLES:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scores.append(fake_probability(Image.fromarray(rgb)))
        if not scores:
            return "No readable frames were available for inference.", ""
        fake_score = sum(scores) / len(scores)
        verdict = "LIKELY DEEPFAKE" if fake_score >= 0.5 else "LIKELY REAL"
        report = (
            "### Deepfake analysis\n"
            f"- **Verdict:** **{verdict}**\n"
            f"- **Fake probability:** **{fake_score:.1%}**\n"
            f"- **Real probability:** **{1 - fake_score:.1%}**\n"
            f"- **Frames analyzed:** {len(scores)}\n"
            f"- **Resolution:** {width} × {height}\n"
            f"- **FPS:** {fps:.2f}\n"
            f"- **Duration:** {duration:.2f} seconds"
        )
        note = (
            "Model: hamzenium/ViT-Deepfake-Classifier. "
            "The score is the mean of sampled frame probabilities, not a forensic certainty. "
            "Performance can change on videos or manipulation methods outside its training distribution."
        )
        return report, note
    except Exception as exc:
        return (
            "Model inference failed. Check dependency installation and network access used to download the model.",
            f"Technical detail: {exc}",
        )
    finally:
        capture.release()

def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Deepfake Video Detector") as demo:
        gr.Markdown(
            "# Deepfake Video Detector\n"
            "Frame-level ViT inference aggregated into a video-level score."
        )
        video = gr.Video(label="Upload a video", type="filepath")
        analyze = gr.Button("Analyze Video", variant="primary")
        result = gr.Markdown()
        note = gr.Markdown()
        analyze.click(fn=analyze_video, inputs=video, outputs=[result, note])
    return demo

if __name__ == "__main__":
    build_app().launch()
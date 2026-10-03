"""Gradio application for temporal deepfake video detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
from PIL import Image

from model import NUM_FRAMES, load_model, predict_clip, sample_frame_indices

def read_clip(video_path: str) -> tuple[list[Image.Image], dict[str, float]]:
    """Read an ordered temporal clip sampled across the video."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video.")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        indices = sample_frame_indices(total_frames)
        frames: list[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {int(index)}.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
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
    """Run one temporal VideoMAE inference over an ordered video clip."""
    if not video_path:
        return "Upload a video to begin.", ""

    path = Path(video_path)
    if not path.exists():
        return "The uploaded video could not be found.", ""

    try:
        # Trigger model download/load before video inference.
        load_model()
        frames, metadata = read_clip(str(path))
        real_score, fake_score = predict_clip(frames)
        verdict = "LIKELY DEEPFAKE" if fake_score >= 0.5 else "LIKELY REAL"

        report = (
            "### Temporal deepfake analysis\n"
            f"- **Verdict:** **{verdict}**\n"
            f"- **Fake probability:** **{fake_score:.1%}**\n"
            f"- **Real probability:** **{real_score:.1%}**\n"
            f"- **Temporal frames:** {NUM_FRAMES}\n"
            f"- **Video frames:** {int(metadata["total_frames"]):,}\n"
            f"- **Resolution:** {int(metadata["width"])} × {int(metadata["height"])}\n"
            f"- **FPS:** {metadata["fps"]:.2f}\n"
            f"- **Duration:** {metadata["duration"]:.2f} seconds"
        )
        note = (
            "Model: SoraExplora/VideoMae. The classifier receives the complete ordered "
            "16-frame sequence together, so its prediction can use temporal relationships "
            "and motion rather than averaging independent frame predictions. The score is "
            "a model output, not a forensic certainty."
        )
        return report, note
    except Exception as exc:
        return (
            "Temporal model inference failed. Check the dependencies and model download.",
            f"Technical detail: {exc}",
        )

def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Temporal Deepfake Video Detector") as demo:
        gr.Markdown(
            "# Temporal Deepfake Video Detector\n"
            "VideoMAE analyzes an ordered sequence of frames to model temporal artifacts."
        )
        video = gr.Video(label="Upload a video", type="filepath")
        analyze = gr.Button("Analyze Video", variant="primary")
        result = gr.Markdown()
        note = gr.Markdown()
        analyze.click(fn=analyze_video, inputs=video, outputs=[result, note])
    return demo

if __name__ == "__main__":
    build_app().launch()
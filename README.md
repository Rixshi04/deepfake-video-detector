# Deepfake Video Analysis Demo

A small OpenCV + Gradio prototype for experimenting with video-frame processing and result visualization.

> **Important:** this repository currently contains a **mock detector**, not a trained deepfake-classification model. It marks demonstration frames as fake so the UI and video-processing pipeline can be tested safely.

## Features
- Upload a video for frame-by-frame processing
- OpenCV-based video decoding
- Gradio web interface
- Detection percentage visualization
- Clear separation between demo logic and a future trained model

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install opencv-python gradio numpy
```

## How the prototype works
1. Open the uploaded video with OpenCV.
2. Iterate through frames.
3. Apply the current demonstration detection rule.
4. Aggregate frame-level results.
5. Display the result through the web UI.

## Limitations
The current logic does **not** learn visual artifacts associated with deepfakes and should not be used as a real authenticity detector.

For a production-quality version, replace the mock rule with a trained model and report precision, recall, F1-score, ROC-AUC, and a confusion matrix on a held-out dataset.

## Suggested next steps
- Add a trained CNN/ViT-based frame classifier.
- Add face detection/cropping before inference.
- Sample frames adaptively instead of processing every frame.
- Batch model inference for better throughput.
- Add automated tests for video loading and inference.
- Add reproducible dependency and model-download instructions.

## License
MIT.
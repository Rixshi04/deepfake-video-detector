# Deepfake Video Analysis Demo

A small **OpenCV + Gradio** prototype for inspecting uploaded videos and sampling frames.

> **Important:** this repository does **not** currently contain a trained deepfake-classification model. The application deliberately reports video properties and frame statistics instead of pretending that a heuristic is a reliable real/fake detector.

## Features

- Upload a video through a Gradio web interface
- Decode videos with OpenCV
- Read resolution, FPS, frame count, and duration
- Sample video frames
- Calculate average sampled-frame brightness
- Clearly distinguish demo analysis from trained deepfake inference

## Project structure

```text
deepfake-video-detector/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### 1. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

From the repository root:

```bash
python app.py
```

Gradio will print a local URL in the terminal. Open it in your browser and upload a video.

## How it works

1. Gradio receives the uploaded video.
2. OpenCV opens the video file.
3. Basic metadata such as FPS, resolution, frame count, and duration is read.
4. A limited number of frames are sampled.
5. Average grayscale brightness is calculated for the sampled frames.
6. The UI reports the analysis and explicitly states that no deepfake prediction is being made.

## Current limitation

This is a **video-analysis/UI prototype**, not a production deepfake detector.

A genuine detector would require a trained model and an evaluation pipeline, for example:

- Face detection and face alignment
- Frame sampling and preprocessing
- CNN or Vision Transformer inference
- Temporal modeling where appropriate
- A labeled train/validation/test dataset
- Precision, recall, F1-score, ROC-AUC, and confusion matrix
- Model/version and dataset documentation

Do not use the current demo's output as evidence that a video is authentic or manipulated.

## License

MIT

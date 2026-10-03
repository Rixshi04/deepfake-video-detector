# Temporal Deepfake Video Detector

A Gradio + OpenCV application that uses a fine-tuned VideoMAE transformer to classify an ordered sequence of video frames as real or deepfake.

> **Important:** This is an ML-based screening tool, not a forensic authenticity guarantee. Model performance can change on videos, compression levels, identities, or manipulation methods outside its training distribution.

## What changed

The detector no longer averages independent frame predictions. It now passes a **16-frame ordered sequence into a temporal VideoMAE model in one inference call**. VideoMAE is designed for spatiotemporal video representations, so the classifier can use relationships across frames rather than treating every frame as an unrelated image. citeturn1search1

## Features

- Upload a video through Gradio
- Uniformly sample 16 ordered frames across the video
- Run temporal VideoMAE inference over the complete sequence
- Produce real/fake probabilities directly from the video classifier
- Display a video-level screening verdict
- Report video metadata

## Temporal model

The application uses `SoraExplora/VideoMae`, a community-published VideoMAE model fine-tuned for binary deepfake video classification. Its model card says it was fine-tuned on a subset of FaceForensics++ using 16 uniformly sampled frames at 224×224 resolution, with real and deepfake classes. The card reports validation accuracy of 88.0%, F1 of 0.742, and AUC of 0.836 on its own held-out validation split. citeturn1view0

Model documentation: urlSoraExplora/VideoMae on Hugging Facehttps://huggingface.co/SoraExplora/VideoMae

The model weights are downloaded and cached automatically by Transformers on first use. They are not committed to this repository.

## How inference works

```text
Video
  ↓
Uniform temporal sampling
  ↓
16 ordered frames
  ↓
VideoMAE processor
  ↓
Temporal VideoMAE transformer
  ↓
Real / Fake logits
  ↓
Softmax probabilities
  ↓
Video-level verdict
```

Unlike the previous frame-averaging implementation, there is no arithmetic average of 16 independent frame probabilities. The model receives the sequence as one video input and its classification head produces the video-level logits.

## Project structure

```text
deepfake-video-detector/
├── app.py
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch installation can be hardware-specific. For NVIDIA GPUs, use the PyTorch build appropriate for your CUDA environment if the default package is not suitable.

### 3. Run

```bash
python app.py
```

Open the local Gradio URL and upload a video.

## Output

```text
Verdict: LIKELY DEEPFAKE
Fake probability: 87.4%
Real probability: 12.6%
Temporal frames: 16
```

The displayed probability is the model's softmax output for the video clip. It is not a calibrated probability that the entire source video is definitively manipulated.

## Limitations

- The model was trained on a subset of FaceForensics++ and may not generalize to unseen manipulation techniques. citeturn1view0
- It samples 16 frames rather than processing every frame.
- Uniform sampling can miss very short-lived manipulations.
- Compression, occlusion, cropping, lighting, and video quality can affect predictions.
- The repository has not independently evaluated the complete application on a held-out dataset.
- The model's reported 88.0% validation accuracy is **not** an accuracy claim for this application. citeturn1view0

For a stronger production/research system, add face-centered preprocessing, multiple temporal clips per video, threshold calibration, and an independent video-level evaluation set.

## License

MIT
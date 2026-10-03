# Deepfake Video Detector

A Gradio + OpenCV application that uses a pretrained Vision Transformer (ViT) deepfake classifier to estimate whether an uploaded video contains manipulated frames.

> **Important:** This is an ML-based screening tool, not a forensic authenticity guarantee. The model is a frame-level classifier trained on the OpenForensics dataset, so performance can change on different datasets, compression levels, identities, or manipulation methods.

## Features

- Upload a video through a Gradio interface
- Sample up to 24 frames from the video
- Run a pretrained ViT deepfake classifier on each sampled frame
- Aggregate frame-level fake probabilities into a video-level score
- Display REAL/DEEPFAKE screening verdict and probability
- Report basic video metadata

## Model

The application uses `hamzenium/ViT-Deepfake-Classifier` from Hugging Face.

- Architecture: Vision Transformer based on `google/vit-base-patch16-224-in21k`
- Task: binary real/fake image classification
- Training data documented by the model card: OpenForensics
- Reported accuracy in the model card: 96.56%
- License reported by the model card: Apache 2.0

The model is downloaded automatically by Transformers on first use and cached locally. The model weights are intentionally not committed to this repository because the checkpoint is large; the repository records the exact model identifier instead.

Model documentation: https://huggingface.co/hamzenium/ViT-Deepfake-Classifier

## How the video score is calculated

1. OpenCV reads the uploaded video.
2. Up to 24 frames are sampled uniformly across the video.
3. Each frame is converted to RGB and passed to the pretrained classifier.
4. The model returns a probability for the fake class.
5. The application calculates the arithmetic mean of the sampled fake probabilities.
6. A score of 50% or higher is displayed as `LIKELY DEEPFAKE`; otherwise it is displayed as `LIKELY REAL`.

The displayed percentage is therefore a **model probability aggregated across sampled frames**, not a calibrated probability that the entire video is definitively fake.

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

PyTorch installation can be hardware-specific. If the standard `pip install` does not provide the appropriate build for your machine, install the matching PyTorch build first and then install the remaining requirements.

### 3. Run the application

```bash
python app.py
```

Open the local Gradio URL printed in the terminal and upload a video.

## Output example

```text
Verdict: LIKELY DEEPFAKE
Fake probability: 87.4%
Real probability: 12.6%
Frames analyzed: 24
```

## Limitations

- The underlying model is a frame-level image classifier; it does not model temporal consistency between frames.
- The model was trained on a particular dataset and may not generalize to every type of deepfake.
- Video compression, lighting, face size, cropping, and unseen manipulation techniques can affect predictions.
- The score is not independently calibrated by this repository.
- A prediction should not be treated as definitive evidence of authenticity or manipulation.
- The application currently analyzes full frames rather than performing dedicated face detection/cropping.

For stronger research use, add a face-detection stage, temporal modeling, a held-out video-level evaluation set, threshold calibration, and reproducible metrics.

## Evaluation

Do not copy the model card's reported accuracy into a claim about this application's accuracy. To make a repository-specific performance claim, evaluate the complete pipeline on a held-out video dataset and report accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

## License

MIT
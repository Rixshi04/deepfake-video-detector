# Temporal Deepfake Video Detector

A Gradio + OpenCV application that uses a fine-tuned VideoMAE transformer to classify an ordered sequence of video frames as real or deepfake.

> **Important:** This is an ML-based screening tool, not a forensic authenticity guarantee. Model performance can change on videos, compression levels, identities, or manipulation methods outside its training distribution.

## What changed

The detector no longer averages independent frame predictions. It now passes a **16-frame ordered sequence into a temporal VideoMAE model in one inference call**. VideoMAE is designed for spatiotemporal video representations, so the classifier can use relationships across frames rather than treating every frame as an unrelated image. citeturn1search1

## Features

- Upload a video through Gradio
- Split the video into up to 5 evenly distributed temporal clips
- Run temporal VideoMAE inference over each 16-frame clip
- Show per-clip temperature-scaled probabilities, score variation, and a calibrated video-level probability
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
Temperature scaling
  ↓
Calibrated video probability
  ↓
Video-level verdict
```

There is no averaging of independent frame predictions. Each 16-frame sequence is passed to VideoMAE as one temporal input. The application converts each clip's real/fake logit with the fitted temperature and computes the final video probability from the mean raw fake-vs-real logit across clips.

## Project structure

```text
deepfake-video-detector/
├── app.py
├── model.py
├── requirements.txt
├── calibration.py
├── calibrate.py
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

For videos long enough to contain multiple clips, the application reports each clip separately. It then shows the mean fake probability, score range, standard deviation, and a confidence summary based on decision strength and cross-clip consistency.


```text
Verdict: LIKELY DEEPFAKE
Fake probability: 87.4%
Real probability: 12.6%
Temporal clips: 5 × 16 frames
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

## Probability calibration

The app does **not** manufacture a confidence score from distance-to-threshold or clip consistency. Instead, it uses **temperature scaling**, a post-hoc calibration method.

For each validation video:

1. Run the same VideoMAE inference used by the app.
2. Convert each clip's raw fake/real probabilities to a fake-vs-real logit:
   `logit = log(p_fake / p_real)`.
3. Average the clip logits to produce one raw video logit.
4. Fit one positive temperature (T) on a labeled validation set by minimizing binary negative log-likelihood:
   `p_calibrated = sigmoid(raw_logit / T)`.
5. Save the fitted temperature to `calibration.json`.
6. The application applies that fixed temperature to new videos.

### Create the calibration file

Prepare a CSV with **one row per labeled validation video**:

```csv
raw_fake_logit,label
1.42,1
-0.83,0
0.37,1
-1.15,0
```

Where `label=0` means real and `label=1` means deepfake. The `raw_fake_logit` values must be generated by the same VideoMAE model and the same multi-clip aggregation used by the app.

Then run:

```bash
python calibrate.py validation_predictions.csv
```

This writes `calibration.json` and reports the validation negative log-likelihood before and after calibration. The generated file is intentionally gitignored because calibration parameters belong to the particular validation dataset/model version.

**Important:** a temperature value cannot be honestly invented from the model's published accuracy/F1/AUC. The repository therefore does not ship a fake calibration parameter. Until you fit `calibration.json` on labeled validation videos, the app will report that calibrated inference is unavailable rather than presenting an uncalibrated number as confidence.

### What the app reports

- **Per-clip temperature-scaled probability:** each clip's raw logit passed through the fitted temperature.
- **Calibrated fake/real probability:** the final video probability after averaging raw clip logits and applying temperature scaling.
- **Predicted-class confidence:** the calibrated probability of whichever class has the higher probability.
- **Clip score variation:** the range between the highest and lowest temperature-scaled clip probabilities.

A calibrated probability means that, on data drawn from the calibration distribution, predictions in a probability bin should approximately match the observed frequency of that outcome. It does **not** mean forensic certainty, and calibration can degrade when the target videos differ substantially from the calibration dataset.


### One-command calibration from labeled videos

You can now build both calibration artifacts directly from a labeled validation set.

Use this structure:

```text
validation_videos/
├── real/
│   ├── real_001.mp4
│   ├── real_002.mp4
│   └── ...
└── fake/
    ├── fake_001.mp4
    ├── fake_002.mp4
    └── ...
```

Then run:

```bash
python build_calibration.py validation_videos
```

The script:

1. Loads the same VideoMAE model as the application.
2. Splits every video into the same maximum of 5 evenly distributed temporal clips.
3. Samples 16 ordered frames per clip.
4. Runs the same temporal classifier on every clip.
5. Converts each clip's real/fake probabilities into a fake-vs-real logit.
6. Averages those clip logits to obtain one raw video logit.
7. Writes `validation_predictions.csv`.
8. Fits temperature scaling on those labeled video logits.
9. Writes `calibration.json`.

Optional output paths:

```bash
python build_calibration.py validation_videos --predictions validation_predictions.csv --calibration calibration.json
```

The script requires at least **20 labeled videos** and both classes. For a meaningful calibration estimate, use a substantially larger, representative validation set when possible and keep it separate from model training data.

The generated `calibration.json` is gitignored because it is specific to the validation distribution and model version.

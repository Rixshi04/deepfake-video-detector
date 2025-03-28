# Deepfake Video Detector

This project is a simple deepfake detection tool that allows users to upload a video and check for deepfake frames. It is built using OpenCV and Gradio to provide a web-based interface.

## Features
- Upload a video file for analysis
- Mock deepfake detection (every second frame is marked as deepfake for demonstration purposes)
- Displays the percentage of detected deepfake frames
- Web-based UI powered by Gradio

## Requirements
Make sure you have Python installed, then install the required dependencies:

```bash
pip install opencv-python gradio numpy
```

## How to Run
Clone this repository and navigate to the project folder:

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
```

Run the script:

```bash
python deepfake_detector.py
```

Once the script runs, it will generate a public Gradio link where you can upload videos for analysis.

## How It Works
1. The script loads the video using OpenCV.
2. It processes each frame and (for demonstration) marks every second frame as a deepfake.
3. The total number of frames and deepfake frames are counted.
4. The result is displayed to the user.

## Example Output
If a video has 100 frames, the output might look like this:
```
Detected Deepfake Frames: 50 out of 100 (50.00%)
```

## Limitations
🚨 **This is NOT a real deepfake detector!** 🚨
- The detection logic is just a mock implementation.
- A real deepfake detector would require a trained deep learning model.

## Future Improvements
- Integrate a real deepfake detection model (e.g., using deep learning)
- Improve frame analysis using facial recognition techniques
- Deploy as a cloud-based service

## Contributing
Pull requests are welcome! Feel free to open issues for feature requests or bug reports.

## License
This project is open-source under the MIT License.


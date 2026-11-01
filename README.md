# Real-Time Object Detection & Captioning

An AI-powered real-time vision assistance system combining object detection, image captioning, and text-to-speech feedback.

## Features

- **Real-Time Object Detection**: Powered by Ultralytics YOLOv8 for fast and accurate bounding-box detection.
- **Image Captioning**: Leverages Salesforce's BLIP (`Salesforce/blip-image-captioning-base`) to generate natural language descriptions of the scene.
- **Voice Feedback (TTS)**: Multi-threaded text-to-speech output using `pyttsx3` with a thread-safe queue to announce descriptions seamlessly without freezing video frames.
- **Camera Utilities**: Built-in scripts to detect available camera indices and test video capture feeds.

## Project Structure

```
real_time_object_captioning/
├── models/
│   └── yolov8n.pt               # YOLOv8 nano pre-trained weights
├── src/
│   ├── captioning.py            # BLIP model test script for image captioning
│   ├── list_cameras.py          # Utility to probe available camera devices
│   ├── object_detection.py      # Real-time YOLOv8 camera detection
│   ├── real_time_assist.py      # Main pipeline: YOLO + BLIP + pyttsx3 speech
│   └── test_camera.py           # Quick webcam testing script
├── static/                      # Sample test images
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shahid-2705/real_time_object_captioning.git
   cd real_time_object_captioning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Check Available Cameras
```bash
python src/list_cameras.py
```

### 2. Run Real-Time Object Detection
```bash
python src/object_detection.py
```

### 3. Run Full Real-Time Assist (Detection + Captioning + Voice)
```bash
python src/real_time_assist.py
```
*Note: In `src/real_time_assist.py` and `src/object_detection.py`, set `CAMERA_INDEX` to match your desired camera feed (default is `0` or `1`).*

## License
MIT

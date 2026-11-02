# AI-Powered Real-Time Visual Assistance System

An asynchronous, modular, and resilient computer vision assistance pipeline combining real-time object detection (**YOLOv8**), natural language image captioning (**Salesforce BLIP**), and auditory feedback (**pyttsx3**), fully optimized for **NVIDIA GPU acceleration (Tensor Cores / FP16)** with automatic CPU fallback.

> [!WARNING]
> **Assistive Tool Disclaimer**  
> This application is intended strictly as an assistive research and visual augmentation tool. It **must not** be relied upon as the sole source of navigation, medical decision-making, or safety-critical awareness in hazardous or unpredictable environments.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Capture["Video Ingestion & Hardware"]
        Cam["CameraManager\n(OpenCV VideoCapture)"] --> Frame["Raw BGR Frame"]
    end

    subgraph Orchestration["VideoProcessor Pipeline"]
        Frame --> YOLO["ObjectDetector\n(YOLOv8 Nano - CUDA/FP16)"]
        Frame --> AsyncQueue["Async Queue\n(maxsize=1, latest frame)"]
        YOLO --> DetRes["Detections + Latency\n(Factual Summary)"]
        DetRes --> HUD["HUD Rendering\n(FPS, Latencies, VRAM, Summaries)"]
        AsyncQueue -.-> CapWorker
        CapWorker --> CapRes["Novel Caption\n(BLIP - CUDA/FP16)"]
        CapRes --> HUD
        HUD --> Display["OpenCV Window Display"]
    end

    subgraph AsyncVision["Asynchronous Vision Worker"]
        CapWorker["AsyncCaptionWorker\n(Dedicated Thread)"]
        CapWorker --> SceneDiff{"Scene Change\n> Threshold?"}
        SceneDiff -- Yes --> BLIP["ImageCaptioner\n(BLIP FP16 autocast)"]
        SceneDiff -- No --> Discard["Skip redundant inference"]
        BLIP --> Dedup{"Text Similarity\n< Threshold?"}
        Dedup -- Novel --> Announce["Trigger Speech Callback"]
    end

    subgraph Audio["Audio Subsystem"]
        Announce --> SpeechEngine["SpeechEngine\n(Dedicated Background Thread)"]
        SpeechEngine --> BoundedQ["Bounded Queue\n(FIFO drop oldest)"]
        BoundedQ --> TTS["pyttsx3 (SAPI5/eSpeak)\nSingle Initialization"]
        TTS --> Speaker["Audio Output"]
    end
```

---

## GPU Acceleration & Optimization Highlights

- **Tensor Core Mixed Precision (FP16)**: BLIP and YOLOv8 natively leverage `torch.float16` and `torch.amp.autocast('cuda')`, halving memory bandwidth and doubling inference speed on modern RTX architecture.
- **CuDNN Benchmark & TF32**: Automatic kernel autotuning (`cudnn.benchmark = True`) and TensorFloat-32 (`allow_tf32 = True`) acceleration enabled by default for matrix multiplication.
- **Kernel Warmup**: Pre-allocates GPU memory and pre-compiles CUDA kernels during startup, eliminating first-frame capture stutter.
- **VRAM Telemetry & Graceful OOM Handling**: Live VRAM allocation tracking on the HUD, with automated cache clearance (`torch.cuda.empty_cache()`) and transparent CPU fallback in out-of-memory scenarios.
- **Non-Blocking Asynchronous Captioning**: Heavy BLIP transformer inference is isolated in a background worker with single-slot latest-frame queuing. Video preview and YOLO detection run continuously at full camera frame rate.

---

## Directory Structure

```
real_time_object_captioning/
├── src/
│   ├── main.py                     # Main CLI application entry point
│   ├── config.py                   # Centralized configuration & argument parsing
│   ├── real_time_assist.py         # Backwards-compatible legacy entry point
│   ├── object_detection.py         # Standalone YOLO detection script (GPU/CPU)
│   ├── captioning.py               # Standalone BLIP testing script (GPU/CPU)
│   ├── test_camera.py              # Camera stream verification utility
│   ├── list_cameras.py             # Available camera device scanner
│   ├── vision/
│   │   ├── detector.py             # YOLOv8 object detector with CUDA/FP16
│   │   ├── captioner.py            # Asynchronous BLIP captioner with FP16/autocast
│   │   └── tracker.py              # Lightweight centroid and object tracking
│   ├── audio/
│   │   └── speech_engine.py        # Dedicated non-blocking pyttsx3 speech worker
│   ├── pipeline/
│   │   └── video_processor.py      # Master video orchestration & HUD rendering
│   └── utils/
│       └── camera.py               # OpenCV camera manager with reconnection
├── tests/                          # Automated unit and integration test suite (21 tests)
├── models/
│   └── yolov8n.pt                  # YOLOv8 nano pre-trained weights
├── assets/                         # Benchmark & test images
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shahid-2705/real_time_object_captioning.git
cd real_time_object_captioning
```

### 2. Virtual Environment Setup
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install GPU-Accelerated PyTorch (Recommended for NVIDIA RTX)

For systems with NVIDIA GPUs (such as RTX 30/40/50 series):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

*For CPU-only fallback, standard PyTorch from `requirements.txt` will be used.*

---

## Execution & Usage

### 1. Run the Full Visual Assistant
```bash
python src/main.py
```
*(or via the legacy script: `python src/real_time_assist.py`)*

### 2. Useful Command-Line Options
```bash
# Run with a specific camera index (e.g. index 1)
python src/main.py --camera-index 1

# Force GPU mode with FP16 acceleration
python src/main.py --device cuda --half

# Run on CPU with FP32
python src/main.py --device cpu --no-half

# Run in silent mode (HUD only, speech disabled)
python src/main.py --no-speech

# Enable object tracking across frames
python src/main.py --enable-tracking

# Enable verbose debug logging
python src/main.py --debug
```

### 3. Interactive Keyboard Controls
| Key | Action |
|:---:|:---|
| `Q` / `ESC` | Exit application cleanly |
| `S` | Toggle voice speech ON / OFF |
| `C` | Force immediate caption generation on current frame |
| `T` | Toggle YOLO object tracking |

---

## Configuration Reference

| Option | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--camera-index` | `int` | `0` | Camera device index |
| `--resolution` | `str` | `640x480` | Frame dimensions (`WIDTHxHEIGHT`) |
| `--confidence` | `float` | `0.35` | Minimum YOLO detection confidence (0.0-1.0) |
| `--caption-interval` | `float` | `3.0` | Minimum seconds between BLIP caption cycles |
| `--min-speech-interval`| `float` | `3.0` | Minimum cooldown between spoken announcements |
| `--device` | `str` | `auto` | Target device (`auto`, `cuda`, `cpu`) |
| `--half` / `--no-half` | flag | `True` | Enable/disable FP16 half-precision for GPU |
| `--speech-rate` | `int` | `160` | TTS words per minute |
| `--speech-volume` | `float` | `1.0` | Audio volume (0.0 - 1.0) |
| `--no-speech` | flag | `False` | Disable audio speech output |
| `--enable-tracking` | flag | `False` | Enable persistent object tracking |
| `--config` | `str` | `None` | Path to optional JSON configuration file |

---

## Measured Performance Benchmarks

| Component | CPU (Intel Core) | NVIDIA GPU (RTX series, FP16) | Optimization Note |
|:---|:---:|:---:|:---|
| **YOLOv8n Object Detection** | 68.5 ms (~14 FPS) | **~5 - 8 ms (~120+ FPS)** | Tensor Cores + FP16 |
| **Salesforce BLIP Captioning** | 858.5 ms (~1.1 FPS) | **~60 - 95 ms (~12+ FPS)** | `amp.autocast('cuda')` |
| **Camera Acquisition & Display** | < 2 ms (30 FPS) | **< 2 ms (30 - 60 FPS)** | OpenCV native buffer |
| **End-to-End Display Latency** | ~71 ms | **~8 - 12 ms** | Non-blocking threading |

---

## Automated Test Suite

Run tests without requiring a physical webcam or downloading model weights:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

```text
Ran 21 tests in 0.961s
OK (All passed)
```

---

## License
MIT

# AI Vehicle Recognition — Drone Surveillance System

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-000000?style=flat-square&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00C853?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-minicpm--v-FFFFFF?style=flat-square)
![MAVLink](https://img.shields.io/badge/MAVLink-Pixhawk%20Ready-FF6F00?style=flat-square)

Real-time aerial **vehicle detection, classification and tracking** for drone surveillance.
Built as an internship project (2026) combining a YOLOv8 vision pipeline, an Ollama-powered
vision language model, and optional autonomous flight control through a Pixhawk flight
controller.

---

## ✨ Features

- 🚁 **Real-time vehicle detection & tracking** — YOLOv8 inference on live/recorded drone video streams
- 🧠 **AI scene description** — detected vehicles are passed to `minicpm-v` via Ollama for natural-language scene analysis
- 🖥️ **Web dashboard** — Flask + Flask-SocketIO UI to upload videos, stream inference live, and monitor results
- 📸 **Automatic frame capture** — detection frames are saved to `captures/` for review and reporting
- 🎯 **Fine-tuned model support** — ships with `yolov8m.pt` and a VisDrone fine-tuned `yolov8m_visdrone.pt` for aerial imagery
- ✈️ **Optional autonomous navigation** — MAVLink integration steers a Pixhawk-controlled drone toward detected targets *(simulation mode without hardware)*

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Detection | YOLOv8 (Ultralytics), OpenCV, NumPy |
| AI Reasoning | Ollama · `minicpm-v` vision language model |
| Backend | Python, Flask, Flask-SocketIO |
| Flight Control | pymavlink · Pixhawk / ArduPilot (optional) |

## 📂 Project Structure

```
AI_Vehicle_Recognition/
├── app.py                  # Main application (detection + web server + MAVLink)
├── download_model.py       # Helper to fetch model weights
├── requirements.txt        # Python dependencies
├── yolov8m.pt              # Base YOLOv8 medium model
├── yolov8m_visdrone.pt     # Fine-tuned on VisDrone aerial dataset
├── templates/
│   └── index.html          # Web dashboard UI
├── videos/                 # Drop .mp4 test files here
├── captures/               # Auto-saved detection frames
└── logs/                   # Runtime logs
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- *(Optional)* [Ollama](https://ollama.com) installed and running for AI scene descriptions
- *(Optional)* Pixhawk + ArduPilot for real flight control

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/ai-vehicle-recognition.git
cd ai-vehicle-recognition

pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Then open the dashboard in your browser and drop an `.mp4` into `videos/` to start inference.

### Optional — enable AI descriptions

```bash
ollama pull minicpm-v
ollama serve
```

> Without Ollama or a Pixhawk, the system automatically runs in **simulation mode** — detection still works fully.

## 📈 How It Works

1. **Input** — drone video (live stream or `.mp4`) enters the pipeline
2. **Detection** — YOLOv8 detects and tracks vehicles frame-by-frame
3. **Reasoning** — key frames are sent to `minicpm-v` for a natural-language situation summary
4. **Action** — frames are saved to `captures/`; with Pixhawk connected, target coordinates are sent via MAVLink for autonomous approach

## 🔮 Future Improvements

- Multi-class threat prioritisation
- Edge deployment (Jetson/ONNX) for onboard inference
- Live FPV stream integration instead of recorded footage

## 👤 Author

**Muhammad Zakwan bin Mohd Suhaimi** — B.Sc. Computer Science (Artificial Intelligence), Universiti Malaya

Internship Project · 2026

---

*Drone flight training conducted outstation at Kahang, Johor and Ulu Bernam, Selangor.*

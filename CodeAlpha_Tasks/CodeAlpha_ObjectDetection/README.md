# 🎯 Task 4 — Object Detection and Tracking
**CodeAlpha AI Internship**

## Features
- **YOLOv8n** (nano) — auto-downloads on first run, no manual setup
- **Centroid-based tracker** — assigns persistent IDs across frames (like SORT)
- Live FPS counter and detection count overlay
- Adjustable confidence threshold slider
- Works with **webcam** or any **video file** (.mp4, .avi, .mov, .mkv)
- Real-time object list sidebar showing detected classes
- Color-coded bounding boxes per tracked ID

## Setup & Run
```bash
pip install -r requirements.txt
python object_detection.py
```
> YOLOv8n model (~6MB) auto-downloads from Ultralytics on first run.

## How It Works
1. Each frame captured from webcam / video is passed to YOLOv8
2. YOLOv8 returns bounding boxes, class names, and confidence scores
3. The centroid tracker computes centroids for each box
4. Hungarian-style minimum-distance assignment links detections across frames
5. Each tracked object gets a unique persistent ID
6. Annotated frame is displayed in real time via Tkinter + PIL

## Tech Stack
- **Python** + **Tkinter** + **Pillow** (GUI)
- **Ultralytics YOLOv8** (object detection)
- **OpenCV** (video capture & drawing)
- **NumPy** (distance matrix, centroid math)
- Custom Centroid Tracker (no external tracking library needed)

## File Structure
```
Task4_ObjectDetection/
├── object_detection.py   # Main application
├── requirements.txt
└── README.md
```

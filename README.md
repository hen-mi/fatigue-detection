# Real-Time Driver Fatigue Detection

A real-time computer vision application that monitors eye movements to detect signs of driver fatigue and drowsiness. Built with Python, OpenCV, and MediaPipe, the system uses facial landmarks and the **Eye Aspect Ratio (EAR)** algorithm to analyze eye closure and trigger alerts when potential fatigue is detected.

## Overview

Driver fatigue is a significant road safety concern. Changes in blinking patterns and prolonged eye closure can indicate reduced alertness and drowsiness.

This project implements an eye-based fatigue detection approach inspired by the research paper *"Eye Aspect Ratio for Real-Time Drowsiness Detection to Improve Driver Safety"* by Christine Dewi et al. (2022).

The application processes webcam footage in real time, detects facial landmarks, calculates the average Eye Aspect Ratio (EAR) for both eyes, and identifies patterns associated with fatigue.

## How It Works

The detection pipeline consists of the following stages:

1. **Video Capture:** OpenCV initializes the webcam and continuously captures video frames.
2. **Face Detection:** MediaPipe Face Mesh identifies the face and maps its facial landmarks.
3. **Eye Landmark Extraction:** Specific landmarks corresponding to the left and right eyes are extracted.
4. **EAR Calculation:** The Eye Aspect Ratio is calculated independently for each eye, and their average is used for analysis.
5. **Fatigue Detection:** The calculated EAR is compared against predefined thresholds to identify eye closure patterns.
6. **Alert System:** When the detection algorithm identifies a fatigue-related pattern, a visual warning is displayed.

## Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is a geometric measurement used to estimate whether an eye is open or closed based on its facial landmarks.

It uses six points around each eye:

- $P_1$ and $P_4$: Horizontal eye boundaries.
- $P_2$, $P_3$, $P_5$, and $P_6$: Vertical eye boundaries.

The EAR is calculated using the following formula:

$$EAR = \frac{\|P_2-P_6\|+\|P_3-P_5\|}{2\|P_1-P_4\|}$$

The average EAR is then calculated from both eyes:

$$EAR_{\text{avg}} = \frac{EAR_{\text{left}}+EAR_{\text{right}}}{2}$$


When the eyes are open, the EAR generally remains relatively stable. When the eyes close, the vertical distances decrease, causing the EAR value to drop.

### Detection Thresholds

The implementation uses four predefined thresholds based on the methodology described by Dewi et al.:

| Threshold | EAR |
|---|---:|
| EAR 1 | 0.180 |
| EAR 2 | 0.200 |
| EAR 3 | 0.225 |
| EAR 4 | 0.250 |

The detector evaluates successive EAR measurements against these thresholds. A sequence of measurements satisfying the detection conditions is used to identify potential fatigue.

## Features

- Real-time webcam video processing.
- Facial landmark detection using MediaPipe Face Mesh.
- Left and right eye landmark extraction.
- Eye Aspect Ratio calculation.
- Blink tracking.
- Threshold-based fatigue detection.
- Visual fatigue alerts.
- Live facial landmark visualization.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video capture and image processing |
| MediaPipe | Facial landmark detection |
| NumPy | Geometric calculations and numerical operations |

## Project Structure

```text
fatigue-detection/
│
├── videosource/          # Video capture module
│
├── facemesh.py           # Facial landmark detection and processing
├── fatiguedetector.py    # EAR calculation and fatigue detection
├── main.py               # Application entry point
├── settings.py           # Application settings
│
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3
- A working webcam
- pip (Python package manager)

### 1. Clone the repository

```bash
git clone https://github.com/hen-mi/fatigue-detection.git

cd fatigue-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the application by running:

```bash
python main.py
```

The application will attempt to initialize the webcam and begin processing video frames.

To close the application, press **Q**.

## Research Reference

This project is an implementation based on the methodology presented in the following research paper:

Dewi, C., Chen, R. C., Yu, H., & Jiang, X. (2022). **Eye Aspect Ratio for Real-Time Drowsiness Detection to Improve Driver Safety.** *Electronics, 11*(19), 3183.

https://doi.org/10.3390/electronics11193183



# AI-Based Emotion Detection with Explainable AI and Interactive Assistant

## Overview

AI-Based Emotion Detection with Explainable AI and Interactive Assistant is a real-time facial emotion recognition system that combines Deep Learning, Explainable AI, and an interactive emotional assistant. The application detects human emotions through a webcam or uploaded image, visualizes model decisions using Grad-CAM heatmaps, and provides personalized emotional support through a voice-enabled assistant.

## Features

* Real-time emotion detection using webcam
* Image upload-based emotion prediction
* Explainable AI using Grad-CAM heatmaps
* Voice-enabled emotional assistant
* Session analytics dashboard
* Automated PDF report generation
* Browser and mobile camera support
* Interactive user-friendly interface

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js
* Web Speech API

### Backend

* Python
* Flask

### Deep Learning & Computer Vision

* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

### Reporting

* ReportLab

## Dataset

The model is trained on the FER2013 (Facial Expression Recognition 2013) dataset consisting of seven emotion classes:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Model Performance

| Model                             | Accuracy |
| --------------------------------- | -------- |
| Improved CNN V2                   | 58.21%   |
| Improved CNN V3                   | 58.80%   |
| MobileNetV2 Fine-Tuned            | 58.10%   |
| Ensemble + Test Time Augmentation | 63.30%   |

## System Architecture

1. Capture image from webcam or upload image.
2. Detect face using OpenCV Haar Cascade.
3. Preprocess facial image.
4. Predict emotion using Ensemble CNN + MobileNetV2.
5. Generate Grad-CAM heatmap for explainability.
6. Display emotion and confidence score.
7. Launch emotional assistant for interaction.
8. Generate analytics and PDF reports.

## Project Structure

```text
emotion-detection-app/
│
├── app.py
├── live_heatmap.py
├── emotion_model_v2.hdf5
├── haarcascade_frontalface_default.xml
│
├── templates/
│   ├── index.html
│   ├── camera.html
│   ├── assistant.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── reports/
├── session_result.json
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/ketannn3/emotion-detection-app.git
cd emotion-detection-app
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

## Explainable AI

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which facial regions contribute most to emotion prediction. This improves model transparency and interpretability.

## Results

* Real-time emotion recognition
* Emotion distribution analytics
* Explainable AI visualizations
* Voice-based emotional assistance
* PDF session reports

## Future Enhancements

* Multilingual support
* Advanced transformer-based emotion models
* Mental health recommendation system
* Cloud deployment
* Mobile application

## Author

**Ketan Mathur**

B.Tech Information Technology
Manipal University Jaipur

GitHub: https://github.com/ketannn3

LinkedIn: https://www.linkedin.com/in/ketan-mathur-3585142aa/

## License

This project is developed for academic and research purposes.

# Drowsiness Detection using YOLOv5

This project focuses on real-time drowsiness detection using a custom-trained YOLOv5 object detection model. The goal is to detect whether a person is **drowsy** or **awake** based on visual cues from a webcam feed. It is especially useful for applications in driver monitoring systems and safety surveillance.

## Problem Statement

Drowsiness is a major cause of accidents, especially in long-haul driving. Timely detection of fatigue or eye closure can help prevent accidents. The project aims to solve this problem by leveraging computer vision to classify a user's state (awake/drowsy) in real time.

## Project Overview

This project uses YOLOv5 (You Only Look Once) — a fast, accurate object detection model — to classify two states:

- `awake`
- `drowsy`

The model is trained on a custom dataset of facial images labeled with these states and is deployed for real-time inference using webcam input.

## Dataset

- Custom dataset consisting of facial images under two categories: `awake` and `drowsy`.
- Images are annotated using YOLO format (bounding boxes + labels).
- Care was taken to balance both classes to prevent model bias.
- Annotation was done using [LabelImg](https://github.com/tzutalin/labelImg) or similar tools.
- The dataset was split into training and validation sets.

## Model

- **YOLOv5s** variant was used for training due to its lightweight and fast inference properties.
- Transfer learning was applied using pre-trained COCO weights.
- Trained using Ultralytics’ YOLOv5 implementation.
- Final model is saved as `best.pt` after monitoring validation loss and mAP.

### Training Details

- Framework: PyTorch
- Base model: YOLOv5s
- Batch size, epochs, and hyperparameters were tuned for accuracy and speed.
- Trained using GPU (if available) or CPU


## Features

* Detects whether a person is drowsy or awake
* Works on live webcam feed
* Fast inference with accurate classification
* Simple integration with other systems for alert mechanisms

## Setup Instructions

1. Clone the YOLOv5 repository:

   ```bash
   git clone https://github.com/ultralytics/yolov5
   ```

2. Install dependencies using pip:

   ```bash
   pip install -r yolov5/requirements.txt
   ```

3. Place the trained model (`best.pt`) inside the `yolov5` directory.

4. Add your webcam detection script or use `detect.py` with modifications.

## Inference & Deployment

* The model can be used in any Python environment with OpenCV and Torch installed.
* For real-time detection, a webcam feed is captured frame-by-frame and passed to the model.
* The model returns predictions with bounding boxes and class labels.

## Limitations

* Accuracy may drop in low-light or occluded environments.
* Works best with frontal face images.
* Dataset quality and size directly affect performance.

## Future Improvements

* Add alert system (beep/buzzer) when drowsiness is detected.
* Improve model with more diverse data (different lighting, angles).
* Extend to include more states (e.g., distracted, phone use).
* Deploy as a mobile or web application.

## Acknowledgements

* [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
* PyTorch
* OpenCV
* LabelImg (for dataset annotation)


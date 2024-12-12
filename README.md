# YOLOv10 Object Detection with Live Webcam Feed

This project demonstrates the use of the YOLOv10 model for real-time object detection through a webcam feed. It processes video frames, runs object detection using a pre-trained YOLO model, and displays bounding boxes and labels for detected objects.

---

## Prerequisites

Before running this project, ensure you have the following installed:

1. **Python 3.7 or later**
2. **Required libraries:**
   - PyTorch
   - OpenCV (`cv2`)
   - Ultralytics YOLO library

---

## Installation

### Step 1: Install Dependencies

Install the required libraries using `pip`:
```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install ultralytics
```

### Step 2: Download the Model

Place the `best10.pt` YOLO model weights file in the same directory as the script. If you don't have this file, train a YOLO model or download a pre-trained one from a trusted source.

---

## How to Run

1. Connect a webcam to your system.
2. Ensure the `best10.pt` file is in the project directory.
3. Run the script:
   ```bash
   python yolov10_detection.py
   ```
4. The webcam feed will appear in a new window, showing detected objects with bounding boxes and labels.
5. Press the `q` key to exit the program.

---

## Code Overview

### Main Components

1. **Preprocessing:**
   - The `preprocess(frame)` function normalizes and converts input frames to a format compatible with the YOLO model.

2. **Prediction:**
   - The `predict(frame)` function uses the YOLO model to predict objects in a given frame.

3. **Webcam Feed:**
   - Captures live video feed using OpenCV.
   - Processes each frame through the YOLO model.
   - Displays bounding boxes and labels on the detected objects.

### Key Functions

- `preprocess(frame)`:
  Converts the input frame to RGB and normalizes pixel values.

- `predict(frame)`:
  Passes the frame to the YOLO model for object detection.

---

## Customization

- **Model Path:** Replace `'best10.pt'` with the path to your YOLO model.
- **Detection Thresholds:** Adjust confidence thresholds and non-maximum suppression in the YOLO library if necessary.
- **Video Source:**
  Change `cv2.VideoCapture(0)` to use a different video source, e.g., a video file.

---

## Troubleshooting

1. **Webcam Not Detected:**
   - Ensure your webcam is connected and recognized by the system.
   - Try changing the argument in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

2. **Missing Dependencies:**
   - Reinstall required libraries with `pip`.

3. **Model Loading Errors:**
   - Verify the `best10.pt` file path and ensure it is compatible with the Ultralytics YOLO library.

---

## License

This project is distributed under the MIT License. Feel free to use and modify the code.

---

## Acknowledgments

- **YOLO (You Only Look Once):** Ultralytics YOLO library for providing the model and framework.
- **OpenCV:** For real-time video capture and processing.


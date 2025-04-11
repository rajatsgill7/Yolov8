# YOLOv8 Human Detection System

This project is a modular implementation of a real-time human detection system using the YOLOv8 model. It features progressive enhancements across five versions (`yolov8_mark1.py` to `yolov8_mark5.py`), including ROI-based detection, fullscreen support, email alerts, motion detection, multi-threaded video streaming, and a live dashboard GUI.

## 📁 File Descriptions

### `yolov8_mark1.py`

> **Basic ROI-Based Human Detection**

- Uses YOLOv8 to detect humans in a video stream.
- Allows the user to draw a Region of Interest (ROI) using the mouse.
- Only humans detected *inside* the ROI are highlighted.
- Fullscreen mode is enabled by default.
- No state saving or advanced interaction features.

---

### `yolov8_mark2.py`

> **State Persistence and Fullscreen Toggling**

- Adds configuration saving/loading for ROI and fullscreen state via `config.json`.
- Introduces keyboard controls:
  - `f`: Toggle fullscreen.
  - `r`: Reset ROI.
- Displays temporary notifications when toggling fullscreen.
- Maintains all features from Mark1.

---

### `yolov8_mark3.py`

> **Email Alerting on Detection**

- Builds on Mark2 with **email alerting** support.
- Sends an email (with snapshot) when a person is detected within the ROI.
- Avoids repeated alerts using an `alert_sent` flag.
- Optional support for IP camera stream (commented example).
- Adds detection timestamping and image capture.
- Secure SMTP login (requires valid email/app password setup).

---

### `yolov8_mark4.py`

> **Motion Detection & Async Streaming**

- Adds **motion detection** to reduce unnecessary model inference.
- Uses frame differencing (`cv2.absdiff`) for activity filtering.
- Adds **multithreaded video streaming** using a `VideoStream` class for improved performance.
- Utilizes GPU acceleration (`torch.device("cuda")`) when available.
- Retains all previous features including email alerting.

---

### `yolov8_mark5.py`

> **Dashboard GUI & Batched Alerts**

- Integrates a **Tkinter dashboard** displaying:
  - Detection status.
  - Total detection count.
  - Timestamp of last detection.
  - Thumbnail of latest detection.
- Adds **batched email alerts**: sends all detections from the past minute in one email.
- Dashboard remains always-on-top for visibility.
- Detection GUI and dashboard run concurrently using threads.
- Continues using motion detection and multithreaded video capture.

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8 (`pip install ultralytics`)
- `pyautogui`, `torch`, `numpy`, `tkinter`, `Pillow`

## ⚙️ Setup

1. Place a YOLOv8 model file (e.g., `yolov8n.pt`) in the root directory.
2. Install the required Python packages.
3. Update the email credentials in Mark3–Mark5 (`SENDER_EMAIL`, `APP_PASSWORD`, `ALERT_EMAIL`).
4. Run any desired script:
   ```bash
   python yolov8_mark5.py
   ```

## 🧪 Notes

- Press `ESC` to exit any version.
- Configure display resolutions according to your monitor using `pyautogui`.
- Ensure permissions are granted for camera and network access if needed.
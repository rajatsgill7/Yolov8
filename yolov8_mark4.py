import cv2
import json
import os
import time
import smtplib
import numpy as np
from email.message import EmailMessage
import pyautogui
from threading import Thread
from ultralytics import YOLO
import torch

# ------------ CONFIG ------------
CONFIG_FILE = 'config.json'
SENDER_EMAIL = "rajatsinghgill31@gmail.com"
APP_PASSWORD = ""  # Replace this with your actual app password
ALERT_EMAIL = "rajatsinghgill31@gmail.com"

# ------------ EMAIL ALERTING ------------
def send_email_alert(subject, body, image_path=None):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = ALERT_EMAIL
        msg.set_content(body)

        if image_path:
            with open(image_path, "rb") as f:
                img_data = f.read()
                msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='snapshot.jpg')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print("[\U0001f4e7] Email alert sent.")
    except Exception as e:
        print(f"[⚠️] Email failed: {e}")

# ------------ CONFIG SAVE/LOAD ------------
def save_config(roi, fullscreen):
    config = {'roi': roi, 'fullscreen': fullscreen}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('roi'), config.get('fullscreen', False)
    return None, False

# ------------ ASYNC VIDEO STREAM ------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, args=()).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ------------ INIT ------------
roi, fullscreen = load_config()
drawing = False
ix, iy = -1, -1
notification_time = 0
alert_sent = False
window_name = "YOLOv8 Human Detection in ROI"
prev_gray = None
model = YOLO('yolov8n.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

screen_width, screen_height = pyautogui.size()
cap = VideoStream(0)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
if fullscreen:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ------------ MOUSE CALLBACK FOR ROI ------------
def draw_roi(event, x, y, flags, param):
    global ix, iy, roi, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))

cv2.setMouseCallback(window_name, draw_roi)

print("🎮 Controls: Draw ROI | [f] Fullscreen toggle | [r] Reset ROI | [ESC] Exit")

# ------------ MAIN LOOP ------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    person_detected = False

    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            if np.sum(thresh) > 100000:  # Adjust this threshold as needed
                results = model(frame, verbose=False)[0]
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    cx = (x1b + x2b) // 2
                    cy = (y1b + y2b) // 2

                    if cls == 0 and x1 <= cx <= x2 and y1 <= cy <= y2:
                        cv2.rectangle(display_frame, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
                        cv2.putText(display_frame, f'Person {conf:.2f}', (x1b, y1b - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        person_detected = True

        prev_gray = gray

        if person_detected and not alert_sent:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(image_path, display_frame)
            send_email_alert("🚨 Person Detected in ROI", f"Detection at {timestamp}", image_path)
            alert_sent = True
        elif not person_detected:
            alert_sent = False

    if time.time() - notification_time < 2:
        msg = "Fullscreen ON" if fullscreen else "Fullscreen OFF"
        cv2.putText(display_frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(1)
    if key == 27:
        save_config(roi, fullscreen)
        break
    elif key == ord('r'):
        roi = None
    elif key == ord('f'):
        fullscreen = not fullscreen
        notification_time = time.time()
        mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)

cap.stop()
cv2.destroyAllWindows()

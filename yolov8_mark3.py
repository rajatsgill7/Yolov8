import cv2
import json
import os
import time
import smtplib
from email.message import EmailMessage
import pyautogui
from ultralytics import YOLO

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
        print("[📧] Email alert sent.")
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

# ------------ INIT ------------
roi, fullscreen = load_config()
drawing = False
ix, iy = -1, -1
notification_time = 0
alert_sent = False
window_name = "YOLOv8 Human Detection in ROI"

model = YOLO('yolov8n.pt')

screen_width, screen_height = pyautogui.size()

# Default camera (webcam)
cap = cv2.VideoCapture(0)

# 📷 Example for switching to IP camera:
# ip_stream_url = "http://192.168.1.2:8080/video"  # <-- replace with your actual IP camera URL
# cap = cv2.VideoCapture(ip_stream_url)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

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

    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        results = model(frame, verbose=False)[0]
        person_detected = False

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

        # Trigger alert once per detection
        if person_detected and not alert_sent:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(image_path, display_frame)
            send_email_alert("🚨 Person Detected in ROI", f"Detection at {timestamp}", image_path)
            alert_sent = True
        elif not person_detected:
            alert_sent = False

    # Notification for fullscreen toggle
    if time.time() - notification_time < 2:
        msg = "Fullscreen ON" if fullscreen else "Fullscreen OFF"
        cv2.putText(display_frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        save_config(roi, fullscreen)
        break
    elif key == ord('r'):
        roi = None
    elif key == ord('f'):
        fullscreen = not fullscreen
        notification_time = time.time()
        mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)

cap.release()
cv2.destroyAllWindows()

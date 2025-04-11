import cv2
import json
import os
import time
import pyautogui
from ultralytics import YOLO

# ----------- Config Management -----------
CONFIG_FILE = 'config.json'

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

# ----------- Initial Setup -----------
# Load saved config (ROI and fullscreen state)
roi, fullscreen = load_config()
drawing = False
ix, iy = -1, -1
notification_time = 0
window_name = "YOLOv8 Human Detection in ROI"

# Load model
model = YOLO('yolov8n.pt')

# Setup video capture
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Create display window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
if fullscreen:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Mouse callback for ROI drawing
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

print("🎮 Draw ROI with mouse | Press 'f' to toggle fullscreen | 'r' to reset ROI | ESC to exit")

# ----------- Main Loop -----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # Draw ROI
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Run detection
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
            cx = (x1b + x2b) // 2
            cy = (y1b + y2b) // 2

            if cls == 0 and x1 <= cx <= x2 and y1 <= cy <= y2:
                cv2.rectangle(display_frame, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
                label = f'Person {conf:.2f}'
                cv2.putText(display_frame, label, (x1b, y1b - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show fullscreen notification
    if time.time() - notification_time < 2:
        msg = "Fullscreen ON" if fullscreen else "Fullscreen OFF"
        cv2.putText(display_frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Show frame
    cv2.imshow(window_name, display_frame)

    # Handle keypresses
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

# ----------- Cleanup -----------
cap.release()
cv2.destroyAllWindows()

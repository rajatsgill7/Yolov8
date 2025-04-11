import cv2
from ultralytics import YOLO
import pyautogui


# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Setup video capture
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width - 250)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height - 250)


cv2.namedWindow("YOLOv8 Human Detection in ROI", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 Human Detection in ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Globals
roi = None
drawing = False
ix, iy = -1, -1

# Mouse callback to draw ROI
def draw_roi(event, x, y, flags, param):
    global ix, iy, roi, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))

# Create window and set mouse callback
cv2.namedWindow("YOLOv8 Human Detection in ROI")
cv2.setMouseCallback("YOLOv8 Human Detection in ROI", draw_roi)

print("Draw ROI with mouse. Press 'r' to reset ROI. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # Draw ROI if defined
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Run inference only if ROI is defined
    if roi:
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
            cx = (x1b + x2b) // 2
            cy = (y1b + y2b) // 2

            # Person class and inside ROI
            if cls == 0 and roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                cv2.rectangle(display_frame, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
                label = f'Person {conf:.2f}'
                cv2.putText(display_frame, label, (x1b, y1b - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Human Detection in ROI", display_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        roi = None  # Reset ROI

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import requests
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2ltwh

# Constants and configuration
SEND_COOLDOWN = 30  # Cooldown time in seconds
CONF_THRESHOLD = 0.75  # Confidence threshold for detection
API_ENDPOINT = "http://your_api_endpoint_here"  # Replace with your API endpoint

# Load the YOLO model and tracker
model = YOLO("yolov8s.pt")
tracker = DeepSort(
    max_iou_distance=0.6,  # Reduced to avoid incorrect associations in close proximity
    max_age=60,  # Extended max_age to allow re-identification after short absences
    n_init=3,  # Retains the default to quickly confirm tracks while filtering noise
    nms_max_overlap=1.0,  # Keeps NMS disabled, since you may need overlapping tracks in crowded scenes
    max_cosine_distance=0.3,  # Slightly increased to tolerate minor appearance changes (lighting, angle)
    gating_only_position=True,  # Only position used for gating, since size changes are less important
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load class names
with open("coco.txt", "r") as f:
    classes = f.read().splitlines()

# Variables for the Region of Interest (ROI)
drawing, ix, iy = False, -1, -1
roi_x1, roi_y1, roi_x2, roi_y2 = -1, -1, -1, -1

# Track IDs and cooldown management
inside_roi_ids = set()
last_send_time = {}
track_history = set()


# API call to send data with specified track ID
# TODO: Move this to TCP so we can communicate back and forth, to ask for screenshot
def send_data(track_id):
    last_send_time[track_id] = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        return

    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg"), "track_id": (None, str(track_id))}

    try:
        response = requests.post(API_ENDPOINT, files=files)
        if response.status_code == 200:
            print(f"Successfully sent data for ID {track_id}.")
        else:
            print(f"Failed to send data for ID {track_id}: {response.status_code}")
    except requests.RequestException as e:
        print(f"An error occurred while sending data: {e}")


# Handlers for person entering, leaving, and re-entering the ROI
def on_new_person_enter(track_id):
    print(f"New person detected entering ROI: ID {track_id}")
    send_data(track_id)


def on_person_leave(track_id):
    print(f"Person exited ROI but is still being tracked: ID {track_id}")


def on_past_person_enter(track_id):
    if time.time() - last_send_time.get(track_id, 0) > SEND_COOLDOWN:
        print(f"Past Track ID {track_id} is back inside ROI.")
        send_data(track_id)


# Mouse callback to define the ROI rectangle
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_x1, roi_y1, roi_x2, roi_y2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_x1, roi_y1, roi_x2, roi_y2 = ix, iy, x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_x1, roi_y1, roi_x2, roi_y2 = ix, iy, x, y


# Setup the main window and mouse callback
cv2.namedWindow("Webcam Stream")
cv2.setMouseCallback("Webcam Stream", draw_rectangle)
print("Press and hold the left mouse button to draw a rectangle. Release to finalize the ROI.")
print("Press 'q' to quit the webcam stream.")

# Main loop for processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Draw ROI if defined
    if roi_x1 != -1 and roi_y1 != -1 and roi_x2 != -1 and roi_y2 != -1:
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # YOLO detections
    results = model(frame, stream=True, verbose=False, conf=CONF_THRESHOLD)
    bbs = [
        (
            xywh2ltwh(box.xywh[0]).cpu().numpy().tolist(),
            box.conf.cpu().numpy().item(),
            box.cls.cpu().numpy().item(),
        )
        for result in results
        for box in result.boxes
        if box.cls.cpu().numpy().item() == 0 and box.xywh[0][2] * box.xywh[0][3] <= frame.shape[0] * frame.shape[1]  # Assuming class 0 is "person"
    ]

    # Update tracker
    tracks = tracker.update_tracks(bbs, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.det_conf is None or int(track.det_class) != 0:
            continue  # Only process confirmed "person" class tracks

        track_id = track.track_id
        left, top, right, bottom = map(int, track.to_ltrb())
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        inside_rectangle = roi_x1 < center_x < roi_x2 and roi_y1 < center_y < roi_y2

        # Draw bounding box and information
        color = (0, 0, 255) if inside_rectangle else (0, 165, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        info_text = (
            f"Class: {classes[int(track.det_class)]} "
            f"ID: {track_id} Conf: {track.det_conf:.2f} "
            f"CD: {np.clip(SEND_COOLDOWN - (time.time() - last_send_time.get(track_id, 0)), 0, SEND_COOLDOWN):.2f}"
        )
        cv2.putText(frame, info_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Manage entry/exit and cooldown resend
        if inside_rectangle:
            if track_id not in track_history and track_id not in inside_roi_ids:
                on_new_person_enter(track_id)
                track_history.add(track_id)
                inside_roi_ids.add(track_id)
            elif track_id in track_history and track_id not in inside_roi_ids:
                on_past_person_enter(track_id)
                inside_roi_ids.add(track_id)
        elif not inside_rectangle and track_id in inside_roi_ids:
            on_person_leave(track_id)
            inside_roi_ids.remove(track_id)

    # Display the frame
    cv2.imshow("Webcam Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

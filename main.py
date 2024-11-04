import cv2
import numpy as np
import json
import time
import socket
import base64
import openai
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2ltwh
import os

# TODO: function for -> i need a greating message to welcome the user with image or not "your at itsec say hello"
# TODO: transcription service for running the transcription model as well (diff process)
# TODO: be able to give image to the openai api for the chatbot to see the image and respond

# Constants and configuration
SEND_COOLDOWN = 30  # Cooldown time in seconds
CONF_THRESHOLD = 0.75  # Confidence threshold for detection
TCP_HOST = "localhost"  # Host for TCP communication
TCP_PORT = 5000  # Port for TCP communication
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set OpenAI API key from environment variable

# Load the YOLO model and tracker
model = YOLO("yolov8s.pt")
tracker = DeepSort(
    max_iou_distance=0.6,
    max_age=60,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    gating_only_position=True,
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

# Global chat history to maintain context across calls
chat_history = []


# Function to send JSON data over TCP with image and other information
def send_data_via_tcp(track_id, text, animation, speaker):
    last_send_time[track_id] = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        return

    # Encode image to JPEG
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_data = img_encoded.tobytes()

    # Prepare JSON data
    data = {"text": text, "animation": animation, "speaker": speaker}
    json_data = json.dumps(data)

    # Set up TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((TCP_HOST, TCP_PORT))
        s.sendall(json_data.encode("utf-8"))
        response = s.recv(1024)
        print("Response from server:", response.decode("utf-8"))


# Function to process OpenAI API requests with optional image handling and chat history tracking
def call_openai_api(prompt, image=None, reset_history=False):
    # Reset chat history if requested
    global chat_history
    if reset_history:
        chat_history = []

    # Model selection
    model = "gpt-4o-mini" if image else "gpt-4-turbo"

    # If image is provided, encode it to base64
    image_data = None
    if image:
        _, buffer = cv2.imencode(".jpg", image)
        image_data = base64.b64encode(buffer).decode("utf-8")

    # Prepare message history with the new prompt
    chat_history.append({"role": "system", "content": "You are Jackson, an AI assistant for the ITSEC Conference."})
    chat_history.append({"role": "user", "content": prompt})
    if image:
        chat_history[-1]["image"] = image_data  # Attach image to the latest user message

    # Call OpenAI API with full chat history
    try:
        response = openai.ChatCompletion.create(model=model, messages=chat_history)

        # Extract and return response content
        assistant_message = response.choices[0].message["content"]

        # Append assistant's response to chat history for continuity
        chat_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None


# Handlers for person entering, leaving, and re-entering the ROI
def on_new_person_enter(track_id):
    print(f"New person detected entering ROI: ID {track_id}")
    text = call_openai_api("Hello, how can I assist you today?", reset_history=True)
    send_data_via_tcp(track_id, text, "walk-in", "speaker1")


def on_person_leave(track_id):
    print(f"Person exited ROI but is still being tracked: ID {track_id}")


def on_past_person_enter(track_id):
    if time.time() - last_send_time.get(track_id, 0) > SEND_COOLDOWN:
        print(f"Past Track ID {track_id} is back inside ROI.")
        send_data_via_tcp(track_id, "Re-entry detected", "walk-in", "speaker2")


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
        if box.cls.cpu().numpy().item() == 0  # Assuming class 0 is "person"
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

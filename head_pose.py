from hpe_module import Network, load_snapshot, genAxis, getAxis, saveImgAxis
import torch
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolo11n.pt")  # Ensure this file is in the working directory

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open the default camera (0 = default webcam, change to 1 or 2 for external camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB (YOLO expects RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model on the frame
    results = model(rgb_frame, conf=0.4, classes=[0, 1, 2, 3, 4, 5, 6, 7])

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Append bounding box in (ltrb, confidence, class_id) format
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get left, top, right, bottom format
        x1, y1, x2, y2 = map(int, ltrb)
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


# file_name = 'group_work'
# input_img_path = "sample_images/" + file_name + ".jpg"
# output_img_path = "sample_images/out/" + file_name + "_bbox.jpg"
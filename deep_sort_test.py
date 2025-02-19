from hpe_module import Network, load_snapshot, genAxis, getAxis, saveImgAxis
import torch
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

model = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=6000)

# Open the default camera (0 = default webcam, change to 1 or 2 for external camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

target_track_id = None
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

    if target_track_id is None:
        for track in tracks:
            if track.is_confirmed():
                target_track_id = track.track_id
                print(f"Target Assigned: {target_track_id}")
                break  # Assign only one person

    # Draw bounding boxes and track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue

        if track.track_id != target_track_id:
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get left, top, right, bottom format
        x1, y1, x2, y2 = map(int, ltrb)

        centroid_x = int(np.ceil((x1 + x2) / 2))
        centroid_y = int(np.ceil((y1 + y2) / 2))

        color = (255, 0, 255)
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=color, thickness=-1)  # Filled circle

        if frame[y1:y2, x1:x2].size != 0:
            frame[y1:y2, x1:x2] = genAxis(frame[y1:y2, x1:x2])

    # Display the frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

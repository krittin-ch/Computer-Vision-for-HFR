from hpe_module import Network, load_snapshot, genAxis, getAxis, saveImgAxis
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


from deepface import DeepFace

path = "fr_data/"

# result = DeepFace.verify(path+"tin_1.jpg", path+"tin_2.jpg")
result = DeepFace.verify(path+"tin_1.jpg", path+"tin_2.jpg", enforce_detection=False, detector_backend='opencv')

print(result)
#DeepFace.find(img_path = ["img1.jpg", "img2.jpg"], db_path = "C:/workspace/my_db") #apply face recognition for multiple identities. this will return a list including pandas dataframe items.


# def get_face_direction(headpose, threshold_pitch=15, threshold_yaw=15, threshold_roll=15):
#     """
#     Determines the direction of the face based on roll, pitch, and yaw.

#     Parameters:
#     - headpose: Tuple (roll, yaw, pitch) in degrees
#     - threshold_pitch: Angle threshold for looking up/down
#     - threshold_yaw: Angle threshold for looking left/right
#     - threshold_roll: Angle threshold for tilting

#     Returns:
#     - String indicating face direction
#     """
#     roll, yaw, pitch = headpose

#     direction = []

#     # Yaw determines left/right
#     if yaw > threshold_yaw:
#         direction.append("Looking Left")
#     elif yaw < -threshold_yaw:
#         direction.append("Looking Right")

#     # Pitch determines up/down
#     if pitch > threshold_pitch:
#         direction.append("Looking Up")
#     elif pitch < -threshold_pitch:
#         direction.append("Looking Down")

#     # Roll determines head tilt
#     if roll > threshold_roll:
#         direction.append("Tilting Right")
#     elif roll < -threshold_roll:
#         direction.append("Tilting Left")

#     if not direction:
#         return "Looking Forward"
    
#     return ", ".join(direction)


# model = YOLO("yolo11n.pt")
# tracker = DeepSort(max_age=6000)

# # Open the default camera (0 = default webcam, change to 1 or 2 for external camera)
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# target_track_id = None
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert frame to RGB (YOLO expects RGB input)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Run YOLO model on the frame
#     results = model(rgb_frame, conf=0.4, classes=[0])
#     # results = model(rgb_frame, conf=0.9)

#     detections = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#             conf = float(box.conf[0])  # Confidence score
#             cls = int(box.cls[0])  # Class ID

#             # Append bounding box in (ltrb, confidence, class_id) format 
#             detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
                
#     # Update DeepSORT tracker
#     tracks = tracker.update_tracks(detections, frame=frame)

#     if target_track_id is None:
#         for track in tracks:
#             if track.is_confirmed():
#                 target_track_id = track.track_id
#                 print(f"Target Assigned: {target_track_id}")
#                 break  # Assign only one person

#     # Draw bounding boxes and track IDs
#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         if track.track_id != target_track_id:
#             continue
        
#         track_id = track.track_id
#         ltrb = track.to_ltrb()  # Get left, top, right, bottom format
#         x1, y1, x2, y2 = map(int, ltrb)

#         centroid_x = int(np.ceil((x1 + x2) / 2))
#         centroid_y = int(np.ceil((y1 + y2) / 2))

#         color = (255, 0, 255)
#         # Draw bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=color, thickness=-1)  # Filled circle

#         # if frame[y1:y2, x1:x2].size != 0:
#         try:
#             frame[y1:y2, x1:x2], hpe_out = getAxis(frame[y1:y2, x1:x2])
#             direction = get_face_direction(hpe_out)
#             cv2.putText(frame, f"DIR {direction}", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         except:
#             pass

#     # Display the frame
#     cv2.imshow('Camera', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()

from hpe_module import Network, load_snapshot, genAxis, getAxis, saveImgAxis
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import time

import face_recognition

path = "database/"  # Change this to your actual image directory
img = path + "target_face.jpg"

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a second sample picture and learn how to recognize it.
tin_image = face_recognition.load_image_file(img)
tin_face_encoding = face_recognition.face_encodings(tin_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    tin_face_encoding,
]
known_face_names = [
    "Krittin Ch",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # Check if there's a match and use the one with the smallest face distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()




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

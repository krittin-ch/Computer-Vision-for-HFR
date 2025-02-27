from hpe_module import getHPAxis, drawAxis
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition
import cv2 
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch 


class DeepHFR():
    def __init__(self, database_folder):

        self.body_model = YOLO("yolo11n.pt")
        self.body_tracker = DeepSort(max_age=1800)
        self.tracks = None
        
        self.target_track_id = {} # {file_name_i: [id_i1, id_i2, ...], file_name_j: [id_j1, id_j2, ...]}
        self.target_encoding = {} # {file_name_i: encoding_i, file_name_j: encoding_j}
        for filename in os.listdir(database_folder):
            file_dir = os.path.join(database_folder, filename)
            if os.path.isfile(file_dir):
                file_name_without_extension = os.path.splitext(filename)[0]
                self.target_track_id[file_name_without_extension] = []

                img = face_recognition.load_image_file(file_dir)
                small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_img)
                self.target_encoding[file_name_without_extension] = face_recognition.face_encodings(rgb_small_img, face_locations)[0]


    def track_body(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.body_model(rgb_frame, conf=0.4, classes=[0])

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  
                cls = int(box.cls[0]) 

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
                    
        self.tracks = self.body_tracker.update_tracks(detections, frame=frame)

    def track_face(self, frame):
        if self.tracks is None: pass

        for track in self.tracks: 
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get left, top, right, bottom format
            body_bbox = map(int, ltrb)

            x1, y1, x2, y2 = body_bbox
            face_bbox, hpe = getHPAxis(frame[y1:y2, x1:x2])

            # self.draw_body(frame, track_id, body_bbox)
            # self.draw_face(frame, face_bbox, hpe)


    

    def draw_body(self, frame, track_id, body_bbox):
            x1, y1, x2, y2 = body_bbox
            centroid_x = int(np.ceil((x1 + x2) / 2))
            centroid_y = int(np.ceil((y1 + y2) / 2))

            color = (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=color, thickness=-1)  # Filled circle

    def draw_face(self, frame, face_bbox, hpe):
        def scale_bbox(self, bbox, scale):
            w = max(bbox[2], bbox[3]) * scale
            x= max(bbox[0] + bbox[2]/2 - w/2,0)
            y= max(bbox[1] + bbox[3]/2 - w/2,0)
            return np.asarray([x,y,w,w],np.int64)
        
        face_images = []
        for i, bbox in enumerate(face_bbox):
            x,y, w,h = scale_bbox(bbox,1.5)
            frame = cv2.rectangle(frame,(x,y), (x+w, y+h),color=(0,0,255),thickness=2) # draw face
            face_img = frame[y:y+h,x:x+w]
            face_images.append(face_img)

        if len(face_images) > 0:
            roll, yaw, pitch = hpe
            for img, r,y,p in zip(face_images, roll,yaw,pitch):
                headpose = [r,y,p]
                drawAxis(img, headpose, size=50)    # draw axes on face
                
 



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

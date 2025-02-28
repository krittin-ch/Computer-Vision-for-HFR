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
    def __init__(self, target_img):

        self.body_model = YOLO("yolo11n.pt")
        self.body_tracker = DeepSort(max_age=1800)
        self.tracks = None

        self.target_track_id = [] # {[id_1, id_2, ...]
        # self.target_encoding = None # encoding from face_recognition

        img = face_recognition.load_image_file(target_img)
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_img)
        self.target_encoding = face_recognition.face_encodings(rgb_small_img, face_locations)[0] # encoding from face_recognition

        """
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
        """

    def track_body(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.body_model(rgb_frame, conf=0.4, classes=[0])

        detections = []
        body_bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  
                cls = int(box.cls[0]) 

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls)) # x, y, w, h
                body_bboxes.append([x1, x2, y1, y2])
        self.tracks = self.body_tracker.update_tracks(detections, frame=frame) # collect all tracks

    def find_target(self, frame):
        if self.tracks is None: 
            return None, None, None, None

        face_bbox = None
        for track in self.tracks: 
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get left, top, right, bottom format
            body_bbox = map(int, ltrb)

            x1, y1, x2, y2 = body_bbox
            f_bbox, hpe = getHPAxis(frame[y1:y2, x1:x2])

            for i, bbox, hpe_i in enumerate(f_bbox, hpe):
                x,y, w,h = self.scale_bbox(bbox,1.5)
                face_encoding = face_recognition.face_encodings(frame[y:y+h,x:x+w])[0]
                matches = face_recognition.compare_faces(self.target_encoding, face_encoding, tolerance=0.3)

                if matches:                
                    self.target_track_id.append(track_id)
                    face_bbox = (x, x+w, y, y+h)
                
                    return body_bbox, face_bbox, hpe_i, track_id
            
            return None, None, None, None


    def run_system(self, frame, if_draw_body=True, if_draw_face=True, if_draw_axis=True):
        body_bbox, face_bbox, hpe, track_id = self.find_target(frame)

        if if_draw_body:
            self.draw_body(frame, track_id, body_bbox)

        if if_draw_face:
            self.draw_face(face_bbox, hpe, if_draw_axis)


    def scale_bbox(self, bbox, scale):
        w = max(bbox[2], bbox[3]) * scale
        x= max(bbox[0] + bbox[2]/2 - w/2,0)
        y= max(bbox[1] + bbox[3]/2 - w/2,0)
        return np.asarray([x,y,w,w],np.int64)

    def draw_body(self, frame, track_id, body_bbox):
        x1, y1, x2, y2 = body_bbox
        centroid_x = int(np.ceil((x1 + x2) / 2))
        centroid_y = int(np.ceil((y1 + y2) / 2))

        color = (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=color, thickness=-1)  # Filled circle

    def draw_face(self, frame, face_bbox, hpe, if_draw_axis=True):
        x1,x2, y1,y2 = face_bbox
        frame = cv2.rectangle(frame,(x1,y1), (x2, y2),color=(0,0,255),thickness=2) # draw face
        face_img = frame[y1:y2,x1:x2]
        
        if if_draw_axis:
            drawAxis(frame[y1:y2,x1:x2], hpe, size=50)    # draw axes on face

            
                
 



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

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
import time
from tqdm import tqdm

class DeepHFR():
    def __init__(self, target_face, target_body_dir=None):

        self.body_model = YOLO("yolo11n.pt")
        self.body_tracker = DeepSort(max_age=1800)
        self.tracks = None

        # self.target_track_id = [] # {[id_1, id_2, ...]
        # self.target_encoding = None # encoding from face_recognition

        # get face encoding
        img = face_recognition.load_image_file(target_face)
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_img)
        self.target_encoding = face_recognition.face_encodings(rgb_small_img, face_locations) # encoding from face_recognition

        # sample images in track id
        self.target_track_id = [] # {[id_1, id_2, ...]
        
        
        if target_body_dir is not None:
            files = [f for f in os.listdir(target_body_dir) if os.path.isfile(os.path.join(target_body_dir, f))]
            for filename in tqdm(files, desc="Processing images", unit="file"):
                file_dir = os.path.join(target_body_dir, filename)
                if os.path.isfile(file_dir):
                    frame = cv2.imread(file_dir, cv2.IMREAD_COLOR_BGR)
                    self.track_body(frame, False)
                
                    for track in self.tracks: 
                        track_id = track.track_id
                        if track_id not in self.target_track_id:
                            self.target_track_id.append(track_id)

        #                 ltrb = track.to_ltrb()  # Get left, top, right, bottom format
        #                 body_bbox = map(int, ltrb)
        #                 self.draw_body(frame, body_bbox, track_id)
                        
        #                 face_locations = face_recognition.face_locations(frame)
        #                 for face_loc in face_locations:
        #                     top, right, bottom, left  = face_loc

        #                     # x1 = left, y1 = top, x2 = right, y2 = bottom 
        #                     self.draw_face(frame, (left, top, right, bottom), None, False)

        #             frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

        #             # Display the frame
        #             cv2.imshow("Tracking Visualization", frame)
        #             key = cv2.waitKey(1)  # Adjust delay if needed

        #             # Press 'q' to exit visualization early
        #             if key & 0xFF == ord('q'):
        #                 break

        # cv2.destroyAllWindows()  

    # track all bodies in a frame
    def track_body(self, frame, verbose=True):
        results = self.body_model(frame, conf=0.75, classes=[0], verbose=verbose)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  
                cls = int(box.cls[0]) 

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls)) # x, y, w, h
        self.tracks = self.body_tracker.update_tracks(detections, frame=frame) # collect all tracks

    def track_face(self, frame, threshold=0.5):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if len(face_encodings) == 0: return None

        face_distances = np.array([])
        matches = np.array([])
        for i, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(self.target_encoding, face_encoding, tolerance=0.5)
            matches = np.append(matches, match)

            # Check if there's a match and use the one with the smallest face distance
            face_distance = face_recognition.face_distance(self.target_encoding, face_encoding)
            face_distances = np.append(face_distances, face_distance)

        best_match_index = np.argmin(face_distances)


        if matches[best_match_index]:
            top, right, bottom, left = face_locations[best_match_index]

            x1 = left
            y1 = top
            x2 = right
            y2 = bottom

            sub_frame = frame[y1:y2, x1:x2]
            best_hpe = getHPAxis(sub_frame)
            # best_bbox = self.pixel_in_frame(
            #     frame, (x1, y1, x2, y2)
            # )

            best_bbox = (left, top, right, bottom)

            return best_bbox, best_hpe  # Return the best-matched face
    
    def find_target(self, frame):
        self.track_body(frame, verbose=False)            # contain all body bbox
        f_out = self.track_face(frame)    # contain only target (face) bbox        

        for track in self.tracks: 

            if track.time_since_update != 0: 
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get left, top, right, bottom format
            body_bbox = self.pixel_in_frame(
                frame, tuple(map(int, ltrb))
            )

            # if tracked without face recognition
            if track_id in self.target_track_id:
                return body_bbox, f_out, track_id
            
            # if the body tracker cannot recogize the body, but face can be recognized; then update tracker id.
            if f_out is not None:
                if self.iou_check(body_bbox, f_out[0]):
                    self.target_track_id.append(track_id)

                    return body_bbox, f_out, track_id

        return None # if cannot track with body and no face to be recognized, the robot cannot follow
            
    def run_system(self, frame, if_draw_body=True, if_draw_face=True, if_draw_axis=True, if_show_update=True):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        prev_target_id = self.target_track_id.copy()

        target_vals = self.find_target(frame)        

        if target_vals is not None:
            body_bbox, f_out, track_id = target_vals

            if if_draw_body and body_bbox is not None:

                self.draw_body(frame, body_bbox, track_id)
            
            if if_draw_face and f_out is not None:
                face_bbox, hpe = f_out
                self.draw_face(frame, face_bbox, hpe, if_draw_axis)

        if if_show_update and len(prev_target_id) != len(self.target_track_id):            
            print("UPDATE ID: ", self.target_track_id)

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def iou_check(self, body_bbox, face_bbox, threshold=0.6):
        x_b1, y_b1, x_b2, y_b2 = body_bbox
        x_f1, y_f1, x_f2, y_f2 = face_bbox

        x_left = max(x_b1, x_f1)
        y_top = max(y_b1, y_f1)
        x_right = min(x_b2, x_f2)
        y_bottom = min(y_b2, y_f2)

        if x_right > x_left and y_bottom > y_top:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
        else:
            intersection_area = 0

        face_area = abs((x_f2 - x_f1) * (y_f2 - y_f1))
        factor = intersection_area / face_area

        if face_area > 0 and factor >= threshold: # the most cases, the factor should be almost 1
            return True
        return False
    
    def pixel_in_frame(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        return (x1, y1, x2, y2)

    def draw_body(self, frame, body_bbox, track_id):
        # ensure bbox to be within frame
        x1, y1, x2, y2 = body_bbox

        centroid_x = int(np.ceil((x1 + x2) / 2))
        centroid_y = int(np.ceil((y1 + y2) / 2))

        color = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
        cv2.circle(frame, (centroid_x, centroid_y), radius=10, color=color, thickness=-1)  # Filled circle

    def draw_face(self, frame, face_bbox, hpe, if_draw_axis=True):
        # top, right, bottom, left = face_bbox # x1 = left, y1 = top, x2 = right, y2 = bottom 
        # frame = cv2.rectangle(frame,(left, top), (right, bottom),color=(0, 0, 255),thickness=4) # draw face

        x1, y1, x2, y2 = face_bbox
        frame = cv2.rectangle(frame,(x1, y1), (x2, y2),color=(0, 0, 255),thickness=4) # draw face
        
        if if_draw_axis and hpe is not None:
            try:
                frame[y1:y2, x1:x2] = \
                    drawAxis(frame[y1:y2, x1:x2], hpe, size=50)    # draw axes on face
            except:
                pass


    # def find_target(self, frame):
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     self.track_body(frame)

    #     if self.tracks is None: 
    #         return None

    #     face_bbox = None
    #     for track in self.tracks: 
    #         track_id = track.track_id
    #         ltrb = track.to_ltrb()  # Get left, top, right, bottom format
    #         body_bbox = map(int, ltrb)

    #         x1, y1, x2, y2 = body_bbox
            
    #         if_process = True
    #         try:
    #             f_bbox, hpe = getHPAxis(frame[y1:y2, x1:x2])

    #         except:
    #             if_process = False
    #             pass

    #         if not if_process or len(f_bbox) == 0: continue

    #         for i, (bbox, hpe_i) in enumerate(zip(f_bbox, hpe)):
    #             x,y, w,h = self.scale_bbox(bbox,1.5)
                
    #             face_encoding = face_recognition.face_encodings(frame[y1:y2, x1:x2][y:y+h, x:x+w])
                
    #             if len(face_encoding) > 0:
    #                 matches = face_recognition.compare_faces(self.target_encoding, face_encoding[0], tolerance=0.3)
    #                 if matches:                
    #                     self.target_track_id.append(track_id)
    #                     face_bbox = (x, x+w, y, y+h)
                    
    #                     return body_bbox, face_bbox, hpe_i, track_id
            
    #         return None


    # def run_system(self, frame, if_draw_body=True, if_draw_face=True, if_draw_axis=True):
    #     target_val = self.find_target(frame)

    #     if target_val is not None:
    #         body_bbox, face_bbox, hpe, track_id = target_val

    #         if if_draw_body:
    #             self.draw_body(frame, track_id, body_bbox)

    #         if if_draw_face:
    #             self.draw_face(face_bbox, hpe, if_draw_axis)

                
 



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

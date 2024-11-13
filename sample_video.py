import numpy as np
import cv2 as cv
from ultralytics import YOLO
from bbox_human import gen_bbox_human, draw_rect
import threading
import queue

class FrameProcessor:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=10)  # Limit the size of the queue
        self.last_bounding_box = None
        self.bbox_lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()

    def process_frames(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:  # Exit signal
                break
            bounding_box = gen_bbox_human(frame)
            with self.bbox_lock:
                self.last_bounding_box = bounding_box
            self.frame_queue.task_done()

    def add_frame(self, frame):
        try:
            self.frame_queue.put(frame, timeout=1)  # Wait for 1 second to put the frame
        except queue.Full:
            print("Frame queue is full, skipping frame.")

    def get_last_bounding_box(self):
        with self.bbox_lock:
            return self.last_bounding_box

# Initialize video capture and frame processor
cap = cv.VideoCapture('v1.MOV')
frame_processor = FrameProcessor()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Add frame to the processing queue
    if frame_count % 10 == 0:  # Process every 10th frame
        frame_processor.add_frame(frame)

    # Get the last bounding box
    last_bounding_box = frame_processor.get_last_bounding_box()
    
    # Draw bounding box if available
    if last_bounding_box is not None:
        frame_with_bbox = draw_rect(frame, last_bounding_box)
        cv.imshow('frame', frame_with_bbox)
    else:
        cv.imshow('frame', frame)

    frame_count += 1
    if cv.waitKey(1) == ord('q'):
        break

# Signal the processing thread to exit
frame_processor.frame_queue.put(None)
frame_processor.processing_thread.join()

cap.release()
cv.destroyAllWindows()
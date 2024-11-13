import numpy as np
import cv2 as cv
from ultralytics import YOLO
from bbox_human import gen_bbox_human

cap = cv.VideoCapture('v1.avi')
 
model = YOLO("yolo11n.pt")

while cap.isOpened():
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    res = gen_bbox_human(frame)
 
    cv.imshow('frame', res)
    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()
<<<<<<< HEAD
from opencv_video_mod import * 

foldername = 'sample_videos/'
filename = 'v3'

show_video(foldername, 'v1', if_show=False, det_rate=10)
# show_video(foldername, 'v3', if_show=False)
=======
import numpy as np
import cv2 as cv
from bbox_human import gen_bbox_human, draw_rect

cap = cv.VideoCapture('v1.MOV')

frame_count = 0
last_bounding_box = None

while cap.isOpened():
    ret, frame = cap.read()

    # if not ret:
    #     print("Can't receive frame (stream end?). Exiting ...")
    #     break

    if frame_count % 5 == 0:
        last_bounding_box = gen_bbox_human(frame)
        frame_with_bbox = draw_rect(frame, last_bounding_box)
        cv.imshow('frame', frame_with_bbox)

    # if last_bounding_box is not None:
    #     frame_with_bbox = draw_rect(frame, last_bounding_box)
    #     cv.imshow('frame', frame_with_bbox)
    # else:
    #     cv.imshow('frame', frame)

    frame_count += 1
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
>>>>>>> 2e8baf463048623f84dc5477f04d2fae51dba050

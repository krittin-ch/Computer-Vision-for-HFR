from ultralytics import YOLO
import cv2
import numpy as np
import os 

# img = cv2.imread("s.jpg")
# cv2.imshow("img1", img)
# cv2.waitKey(0)

model = YOLO("yolo11n.pt")

res = model.predict(
<<<<<<< HEAD
    "v1.mp4", 
    save=True, 
    save_txt=True, 
    show_conf=False,
    conf = 0.4,
    classes=[0, 1, 2, 3, 4, 5, 6, 7],
    project="output", name="o_1"
)

# print(res)
=======
    img, 
    save=False, 
    save_txt=False, 
    show_conf=False,
    conf = 0.6,
    classes=[0],
    # project="output", name="o_1"
)

print(res)
>>>>>>> 2e8baf463048623f84dc5477f04d2fae51dba050

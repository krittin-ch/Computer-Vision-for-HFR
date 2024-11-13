from ultralytics import YOLO
import cv2
import numpy as np
import os 

img = cv2.imread("s.jpg")
# cv2.imshow("img1", img)
# cv2.waitKey(0)

model = YOLO("yolo11n.pt")

res = model.predict(
    img, 
    save=False, 
    save_txt=False, 
    show_conf=False,
    conf = 0.6,
    classes=[0],
    # project="output", name="o_1"
)

print(res)
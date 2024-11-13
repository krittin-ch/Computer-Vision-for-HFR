import sys 
import os
import logging
from ultralytics import YOLO
import cv2
import numpy as np

# logging.basicConfig(level=logging.CRITICAL)
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

model = YOLO("yolo11n.pt", verbose=False)

def gen_bbox_human(img, acc=0.5):
    results = model(img)

    bbox = results[0].boxes
    cls_res = bbox.cls.cpu().detach().numpy()

    mark_people = np.where(cls_res == 0)
    acc_res = bbox.conf.cpu().detach().numpy()[mark_people]

    mark_acc = np.where(acc_res > acc)

    acc_res = acc_res[mark_acc]

    pos_res = bbox.xyxy.cpu().detach().numpy()[mark_people][mark_acc]

    # img = draw_rect(img, pos_res)

    return pos_res

def draw_rect(img, pos_res):
    color = (255, 0, 255)
    thickness = 4
    
    for pos in pos_res:
        x1, y1, x2, y2 = pos

        start_point = (int(np.ceil(x1)), int(np.ceil(y1)))
        end_point = (int(np.ceil(x2)), int(np.ceil(y2)))


        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    
    return img
# img = cv2.imread("bus.jpg")

# res = gen_bbox_human(img)

# cv2.imshow("img1", res)
# cv2.waitKey(0)
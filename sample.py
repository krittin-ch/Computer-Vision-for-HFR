from ultralytics import YOLO
import cv2
import numpy as np

img = cv2.imread("s.jpg")
# cv2.imshow("img1", img)
# cv2.waitKey(0)

model = YOLO("yolo11n.pt")

results = model(img)

# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk

"""
names: {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
"""

x = results[0].boxes
cls_res = x.cls.cpu().detach().numpy()

mark_people = np.where(cls_res == 0)
acc_res = x.conf.cpu().detach().numpy()[mark_people]

mark_acc = np.where(acc_res > 0.7)

acc_res = acc_res[mark_acc]

pos_res = x.xyxy.cpu().detach().numpy()[mark_people][mark_acc]

color = (0, 0, 0)

res = np.copy(img)

for pos in pos_res:
    x1, y1, x2, y2 = pos

    start_point = (int(np.ceil(x1)), int(np.ceil(y1)))
    end_point = (int(np.ceil(x2)), int(np.ceil(y2)))

    thickness = 3

    res = cv2.rectangle(res, start_point, end_point, color, thickness)

cv2.imshow("res", res) 
cv2.waitKey(0)
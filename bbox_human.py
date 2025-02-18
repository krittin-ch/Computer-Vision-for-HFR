from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Optional

model = YOLO("yolo11n.pt", verbose=False)

<<<<<<< HEAD

# def gen_bbox_human(img, acc=0.8):

#     results = model.predict(
#         img, 
#         save=False, 
#         save_txt=False, 
#         show_conf=False,
#         conf = acc,
#         classes=[0],
#         project="output", name="o_1", exist_ok=True
#     )

#     bbox = results[0].boxes
#     pos_res = bbox.xyxy.cpu().detach().numpy()

#     print(type(pos_res[0]))
#     return pos_res

# def draw_rect(img, pos_res, if_centroid=False, if_coords=False):
#     color = (255, 0, 255)
#     thickness = 1
    
#     for pos in pos_res:
#         x1, y1, x2, y2 = pos

#         start_point = (int(np.ceil(x1)), int(np.ceil(y1)))
#         end_point = (int(np.ceil(x2)), int(np.ceil(y2)))

#         img = cv2.rectangle(img, start_point, end_point, color, thickness)

#         centroid_x = int(np.ceil((x1 + x2) / 2))
#         centroid_y = int(np.ceil((y1 + y2) / 2))
        
#         if if_centroid:
#             img = cv2.circle(img, (centroid_x, centroid_y), radius=5, color=color, thickness=-1)  # Filled circle

#         if if_coords:
#             text = f"({centroid_x}, {centroid_y})"
#             text_position = (int(np.ceil((centroid_x))), int(np.ceil((y1 - 5))))

#             img = cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
#                               fontScale=0.5, color=color, thickness=2, lineType=cv2.LINE_AA)
#     return img


def gen_bbox_human(img: np.ndarray, acc: float=0.8) -> np.array:
    '''
    inputs:
        img: (H, W, 3) rgb numpy array
        acc: accuracy threshold 
        (if an object is detected with an accuracy less than acc, 
        an object will not be considered)

    output:
        pos_res: position result 
        (numpy array of [[x1, x2, y1, y2]]: top left and bottom right (or top right and bottom left, I guess))
        (if cannot detect, return as [])
    '''
    results = model.predict(
        img, 
        save=False,     
        save_txt=False, 
        show_conf=False,
        conf = acc,
        classes=[0],        # class '0' is for human. Therefore, detecting only human
=======
def gen_bbox_human(img, acc=0.6):

    results = model.predict(
        img, 
        save=True, 
        save_txt=False, 
        show_conf=False,
        conf = acc,
        classes=[0],
>>>>>>> 2e8baf463048623f84dc5477f04d2fae51dba050
        project="output", name="o_1", exist_ok=True
    )

    bbox = results[0].boxes
<<<<<<< HEAD
    # in case, the model is run on GPU, 
    # we need to convert the data into CPU for visualization
    pos_res = bbox.xyxy.cpu().detach().numpy()  
=======
    # cls_res = bbox.cls.cpu().detach().numpy()
    # acc_res = bbox.conf.cpu().detach().numpy()
    pos_res = bbox.xyxy.cpu().detach().numpy()
    # img = draw_rect(img, pos_res)
>>>>>>> 2e8baf463048623f84dc5477f04d2fae51dba050

    return pos_res

def draw_rect(img: np.ndarray, pos_res: np.ndarray, if_centroid: bool=True, if_coords: bool=True) \
    -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    '''
    inputs:
        img: (H, W, 3) rgb numpy array
        pos_res: numpy array of [[x1, x2, y1, y2]] or [] from gen_bbox_human

    output:
        img: (H, W, 3) rgb numpy array with bounding box and centroid
        centroid_res: (centroid_x, centroid_y) tuple of int
    '''
    color = (255, 0, 255)
    thickness = 1
    
    centroid_res = None
    for pos in pos_res:
        x1, y1, x2, y2 = pos

        start_point = (int(np.ceil(x1)), int(np.ceil(y1)))
        end_point = (int(np.ceil(x2)), int(np.ceil(y2)))

        img = cv2.rectangle(img, start_point, end_point, color, thickness)
<<<<<<< HEAD
=======
    
    return img

# img = cv2.imread("bus.jpg")
>>>>>>> 2e8baf463048623f84dc5477f04d2fae51dba050

        centroid_x = int(np.ceil((x1 + x2) / 2))
        centroid_y = int(np.ceil((y1 + y2) / 2))

        centroid_res = (centroid_x, centroid_y)
        
        if if_centroid:
            img = cv2.circle(img, (centroid_x, centroid_y), radius=5, color=color, thickness=-1)  # Filled circle

        if if_coords:
            text = f"({centroid_x}, {centroid_y})"
            text_position = (int(np.ceil((centroid_x))), int(np.ceil((y1 - 5))))

            img = cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=0.5, color=color, thickness=2, lineType=cv2.LINE_AA)
            
    return img, centroid_res
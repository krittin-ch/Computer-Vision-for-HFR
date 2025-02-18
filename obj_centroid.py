from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, Optional

# use case is below

# load model
# ultralytics website: https://docs.ultralytics.com/models/yolo11/#performance-metrics
# model: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
model = YOLO("yolo11n.pt", verbose=False)

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
        project="output", name="o_1", exist_ok=True
    )

    bbox = results[0].boxes
    # in case, the model is run on GPU, 
    # we need to convert the data into CPU for visualization
    pos_res = bbox.xyxy.cpu().detach().numpy()  

    return pos_res

def draw_rect(img: np.ndarray, pos_res: np.ndarray, if_centroid: bool=True, if_coords: bool=True) \
    -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    '''
    inputs:
        img: (H, W, 3) rgb numpy array
        pos_res: numpy array of [[x1, x2, y1, y2]] or [] from gen_bbox_human
        if_centroid: draw centroid on the image or not (even if it's false, the program still returns centroid, but not draws)
        if_coords: add coordinates (x, y) text on the image

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



# Replace with your image path. 
# Normally, you directly retrieve data from your software as numpy, 
# so you can ignore this line.
img = cv2.imread('img.jpg') 

pos_res = gen_bbox_human(img, 0.6) # add img, and acc=0.6
img, centroid_res = draw_rect(img, pos_res, if_centroid=True, if_coords=False) # put centroid on the image, but no coordinates

cen_x, cen_y = centroid_res # retrieve x and y coordinates from centroid information
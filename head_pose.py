from hpe_module import Network, load_snapshot, genAxis, getAxis
import torch
import cv2

file_name = 'group_work'
input_img_path = "sample_images/" + file_name + ".jpg"
output_img_path = "sample_images/out/" + file_name + "_bbox.jpg"

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo11n.pt")

res = model.predict(
    input_img_path,  
    save=True, 
    save_txt=True, 
    show_conf=False,
    conf = 0.4,
    classes=[0, 1, 2, 3, 4, 5, 6, 7],
    project="output", name="o_1"
)

print(res)

# tracker = DeepSort(max_age=5)
# tracks = tracker.update_tracks(bbs, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
# for track in tracks:
#     if not track.is_confirmed():
#         continue
#     track_id = track.track_id
#     ltrb = track.to_ltrb()


genAxis(input_img_path, output_img_path)

frame = cv2.imread(input_img_path)
x = getAxis(frame)



# Get the result
# roll, yaw, pitch = model(img_tensor)

# print(roll, yaw, pitch)

# Draw the results on the image
# cv2.imshow("Image", model.draw(image, poses, lms, bbox, draw_face=True, draw_person=True, draw_axis=True))
# cv2.imwrite("sample_images/output/" + file_name, model.draw(image, poses, lms, bbox, draw_face=True, draw_person=True, draw_axis=True))
# cv2.waitKey(0)

# from head_pose_module import SimplePose
# import cv2

# model = SimplePose(model_type="svr")
# model.load("best_model_svr_23_01_24_17")

# # Load an image from the given path
# file_name = 'city_person.jpg'
# image = cv2.imread("sample_images/" + file_name)

# # Flip the image horizontally for a later selfie-view display
# # Also convert the color space from BGR to RGB
# image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

# # Get the result
# poses, lms, bbox = model.predict(image)

# # Draw the results on the image
# # cv2.imshow("Image", model.draw(image, poses, lms, bbox, draw_face=True, draw_person=True, draw_axis=True))
# cv2.imwrite("sample_images/output/" + file_name, model.draw(image, poses, lms, bbox, draw_face=True, draw_person=True, draw_axis=True))
# # cv2.waitKey(0)
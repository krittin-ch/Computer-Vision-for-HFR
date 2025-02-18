from head_pose_module import SimplePose
import cv2

model = SimplePose(model_type="svr")
model.load("best_model_svr_23_01_24_17")

# Load an image from the given path
file_name = 'ex_5.jpg'
image = cv2.imread("sample_images/" + file_name)

# Flip the image horizontally for a later selfie-view display
# Also convert the color space from BGR to RGB
image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

# Get the result
poses, lms, bbox = model.predict(image)

# Draw the results on the image
# cv2.imshow("Image", model.draw(image, poses, lms, bbox, draw_face=True, draw_person=True, draw_axis=True))
cv2.imwrite("sample_images/output/" + file_name, model.draw(image, poses, lms, bbox, draw_face=True, draw_person=True, draw_axis=True))
# cv2.waitKey(0)
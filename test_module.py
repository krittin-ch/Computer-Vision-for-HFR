from deep_sort_module import DeepHFR
import cv2 
import time

target_body_dir = "database/body_img/"
target_face = "database/target_face.jpg"

deep_hfr = DeepHFR(target_face, target_body_dir)

print("SHOW INITIAL TARGET IDs")
print(deep_hfr.target_track_id)
print()

time.sleep(3)

print("---------- START DETECTION ----------")

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    frame = deep_hfr.run_system(
        frame, 
        if_draw_body=True,
        if_draw_face=True,
        if_draw_axis=True,
        if_show_update=True
    )

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("---------- END DETECTION ----------")

video_capture.release()
cv2.destroyAllWindows()
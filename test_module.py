from deep_sort_module import DeepHFR
import cv2 

target_img = "fr_data/tin_1.jpg"

deep_hfr = DeepHFR(target_img)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    deep_hfr.run_system(frame)

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

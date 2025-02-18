import cv2 as cv
from bbox_human import gen_bbox_human, draw_rect
import os

def show_video(foldername, filename, det_rate=5, if_show=True, if_record=True):
    video_path = os.path.join(foldername, f"{filename}.mp4")
    cap = cv.VideoCapture(video_path)

    # Validate if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    # Prepare VideoWriter if recording is enabled
    if if_record:
        output_folder = os.path.join(foldername, 'output')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{filename}_with_bboxes.mp4")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = None

    frame_count = 0
    last_bounding_box = None

    while cap.isOpened():
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret or frame is None:
            print("End of video stream or invalid frame. Exiting...")
            break

        # Generate bounding boxes every `det_rate` frames
        if frame_count % det_rate == 0:
            try:
                last_bounding_box = gen_bbox_human(frame, 0.5)  # Replace `gen_bbox_human` with your detection function
            except Exception as e:
                print(f"Error during bounding box generation: {e}")
                last_bounding_box = None

        # Draw bounding boxes on the frame
        if last_bounding_box is not None:
            frame_with_bbox, _ = draw_rect(frame, last_bounding_box, True)
        else:
            frame_with_bbox = frame

        # Save the frame to the output video
        if if_record and out is not None:
            out.write(frame_with_bbox)

        # Display the frame (if enabled)
        if if_show:
            cv.imshow('Processed Frame', frame_with_bbox)

        frame_count += 1

        # Exit on 'q' key press
        if cv.waitKey(1) == ord('q'):
            print("Exiting due to user input...")
            break

    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv.destroyAllWindows()
    print("Video processing completed.")

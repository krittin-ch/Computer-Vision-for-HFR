import cv2

# Define a callback function to capture mouse events
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # When the mouse is moved
        img_copy = img.copy()  # Make a copy of the image to overlay the text
        text = f"X: {x}, Y: {y}"
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Image', img_copy)

# Load your image
img = cv2.imread('p2.jpg')  # Replace with your image path

# Display the image in a window
cv2.imshow('Image', img)

# Set the mouse callback function
cv2.setMouseCallback('Image', show_coordinates)

# Wait until a key is pressed, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

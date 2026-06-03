import cv2
import matplotlib.pyplot as plt

video_path = './projeto/videos/019_Maria_Elenira_pre.mov'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Read the first frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame from BGR to RGB (Matplotlib expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the first frame
        plt.imshow(frame_rgb)
        plt.title("First Frame of the Video")
        plt.axis('off') # Hide axes for better image display
        plt.show()
    else:
        print("Error: Could not read the first frame.")

    # Release the video capture object
    cap.release()
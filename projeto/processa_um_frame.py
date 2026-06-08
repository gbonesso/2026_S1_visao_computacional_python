import cv2
import csv
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from pre_processamento import get_iris_mask, preprocess_frame
from teste2 import make_panel

video_path = './projeto/videos/15_Bruna_PR_Pre_2025-04-20T09_53_50_734327_output.MP4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Vai para o frame 65
cap.set(cv2.CAP_PROP_POS_FRAMES, 65)
# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    ret, frame = cap.read()

    if ret:
        # Convert the frame from BGR to RGB (Matplotlib expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        panel_width = frame_width // 2
        panel_height = frame_height

        mid = frame_rgb.shape[1] // 2
        left_eye = frame_rgb[:, :mid]
        right_eye = frame_rgb[:, mid:]

        top_left = make_panel(left_eye, 'Left eye', (panel_width, panel_height))
        top_right = make_panel(right_eye, 'Right eye', (panel_width, panel_height))

        left_edges = preprocess_frame(left_eye)
        right_edges = preprocess_frame(right_eye)

        left_iris_mask, left_eye_iris_bottom, left_eye_iris_top, left_img_clahe = get_iris_mask(left_eye)
        right_iris_mask, right_eye_iris_bottom, right_eye_iris_top, right_img_clahe = get_iris_mask(right_eye)

        left_opening = max(0.0, float(left_eye_iris_bottom) - float(left_eye_iris_top))
        right_opening = max(0.0, float(right_eye_iris_bottom) - float(right_eye_iris_top))

        # Display the first frame
        plt.imshow(frame_rgb)
        plt.title("Frame 65 of the Video")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display CLAHE images for left and right eyes side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        if left_img_clahe is not None:
            if getattr(left_img_clahe, 'ndim', 2) == 2:
                axs[0].imshow(left_img_clahe, cmap='gray')
            else:
                axs[0].imshow(left_img_clahe)
            axs[0].set_title('Left img_clahe')
            axs[0].axis('off')
        else:
            axs[0].set_title('Left img_clahe (None)')
            axs[0].axis('off')

        if right_img_clahe is not None:
            if getattr(right_img_clahe, 'ndim', 2) == 2:
                axs[1].imshow(right_img_clahe, cmap='gray')
            else:
                axs[1].imshow(right_img_clahe)
            axs[1].set_title('Right img_clahe')
            axs[1].axis('off')
        else:
            axs[1].set_title('Right img_clahe (None)')
            axs[1].axis('off')

        plt.show()
    else:
        print("Error: Could not read the frame.")

    # Release the video capture object
    cap.release()
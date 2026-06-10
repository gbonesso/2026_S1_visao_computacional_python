import cv2
import csv
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from pre_processamento import get_iris_mask, preprocess_frame
from teste2 import make_panel


def draw_dotted_hline(image, y, color, dash_length=12, gap_length=8, thickness=1):
    height, width = image.shape[:2]
    y = int(round(y))
    if y < 0 or y >= height:
        return image

    x = 0
    while x < width:
        x_end = min(x + dash_length, width)
        cv2.line(image, (x, y), (x_end, y), color, thickness, cv2.LINE_AA)
        x += dash_length + gap_length

    return image


def annotate_iris_panel(panel, iris_mask, iris_top, iris_bottom):
    mask_height = iris_mask.shape[0]
    panel_height = panel.shape[0]

    if mask_height > 0:
        top_y = (float(iris_top) / mask_height) * panel_height
        bottom_y = (float(iris_bottom) / mask_height) * panel_height

        draw_dotted_hline(panel, top_y, (0, 255, 255))
        draw_dotted_hline(panel, bottom_y, (255, 0, 255))

        opening = max(0.0, float(iris_bottom) - float(iris_top))
        cv2.putText(
            panel,
            f'Opening: {opening:.1f}',
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return panel

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

        left_iris_mask, left_eye_iris_bottom, left_eye_iris_top, left_img_clahe, left_img_gray_blurred, \
            left_img_gray_upscaled, left_iris_mask_threshold, left_iris_mask_dilated, left_iris_mask_morph_close, \
            left_iris_mask_morph_open = get_iris_mask(left_eye)
        right_iris_mask, right_eye_iris_bottom, right_eye_iris_top, right_img_clahe, right_img_gray_blurred, \
            right_img_gray_upscaled, right_iris_mask_threshold, right_iris_mask_dilated, right_iris_mask_morph_close, \
            right_iris_mask_morph_open = get_iris_mask(right_eye)

        left_opening = max(0.0, float(left_eye_iris_bottom) - float(left_eye_iris_top))
        right_opening = max(0.0, float(right_eye_iris_bottom) - float(right_eye_iris_top))

        # Display the first frame
        plt.imshow(frame_rgb)
        plt.title("Frame 65 of the Video")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left eye with CLAHE applied
        plt.imshow(left_img_clahe, cmap='gray')
        plt.title("CLAHE")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left eye with Gaussian Blur applied
        plt.imshow(left_img_gray_blurred, cmap='gray')
        plt.title("Gaussian Blur")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left eye with threshold mask applied        
        left_img_gray_upscaled_rgb = cv2.cvtColor(left_img_gray_upscaled, cv2.COLOR_GRAY2RGB)
        left_iris_overlay = left_img_gray_upscaled_rgb.copy()
        left_iris_overlay[left_iris_mask_threshold > 0] = [255, 0, 0]
        left_iris_overlay = cv2.addWeighted(left_img_gray_upscaled_rgb, 0.7, left_iris_overlay, 0.3, 0)

        plt.imshow(left_iris_overlay)
        plt.title("Upscaled + Iris Mask Threshold")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left eye with dilated iris mask applied
        left_iris_dilated_overlay = left_img_gray_upscaled_rgb.copy()
        left_iris_dilated_overlay[left_iris_mask_dilated > 0] = [255, 0, 0]
        left_iris_dilated_overlay = cv2.addWeighted(left_img_gray_upscaled_rgb, 0.7, left_iris_dilated_overlay, 0.3, 0)

        plt.imshow(left_iris_dilated_overlay)
        plt.title("Upscaled + Iris Mask Dilated")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left eye with morphologically closed iris mask applied
        left_iris_morph_close_overlay = left_img_gray_upscaled_rgb.copy()
        left_iris_morph_close_overlay[left_iris_mask_morph_close > 0] = [255, 0, 0]
        left_iris_morph_close_overlay = cv2.addWeighted(left_img_gray_upscaled_rgb, 0.7, left_iris_morph_close_overlay, 0.3, 0)

        plt.imshow(left_iris_morph_close_overlay)
        plt.title("Upscaled + Iris Mask Morphologically Closed")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left eye with morphologically opened iris mask applied
        left_iris_morph_open_overlay = left_img_gray_upscaled_rgb.copy()
        left_iris_morph_open_overlay[left_iris_mask_morph_open > 0] = [255, 0, 0]
        left_iris_morph_open_overlay = cv2.addWeighted(left_img_gray_upscaled_rgb, 0.7, left_iris_morph_open_overlay, 0.3, 0)

        plt.imshow(left_iris_morph_open_overlay)
        plt.title("Upscaled + Iris Mask Morphologically Opened")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left eye with final iris mask applied
        left_iris_final_overlay = left_img_gray_upscaled_rgb.copy()
        left_iris_final_overlay[left_iris_mask > 0] = [255, 0, 0]
        left_iris_final_overlay = cv2.addWeighted(left_img_gray_upscaled_rgb, 0.7, left_iris_final_overlay, 0.3, 0)

        plt.imshow(left_iris_final_overlay)
        plt.title("Upscaled + Iris Mask Final")
        plt.axis('off') # Hide axes for better image display
        plt.show()

        # Display the left iris mask with annotations like teste2.py
        left_iris_mask_panel = make_panel(
            left_iris_final_overlay, 
            'Left iris mask', 
            (left_iris_final_overlay.shape[1], left_iris_final_overlay.shape[0])
        )
        left_iris_mask_panel = annotate_iris_panel(
            left_iris_mask_panel,
            left_iris_mask,
            left_eye_iris_top,
            left_eye_iris_bottom,
        )

        plt.imshow(cv2.cvtColor(left_iris_mask_panel, cv2.COLOR_BGR2RGB))
        plt.title("Left iris mask annotated")
        plt.axis('off')
        plt.show()

        
        

    else:
        print("Error: Could not read the frame.")

    # Release the video capture object
    cap.release()
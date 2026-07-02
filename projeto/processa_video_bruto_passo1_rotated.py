import cv2
import csv
import numpy as np
from pathlib import Path

from pre_processamento import get_iris_mask, preprocess_frame
from processa_video_bruto_passo1 import annotate_iris_panel, make_panel, overlay_iris_mask
from pos_processamento import get_angle_from_mask, get_top_bottom_opening_from_mask, rotate_image_and_mask

video_path = './projeto/videos2/30_Nicolas_PR_Pre_2025-04-19T20_24_11_016457_output.MP4'
output_path = './projeto/videos2/30_Nicolas_PR_Pre_2025-04-19T20_24_11_016457_montage_rotated.mp4'
overlay_output_path = './projeto/videos2/30_Nicolas_PR_Pre_2025-04-19T20_24_11_016457_output_iris_overlay_rotated.mp4'
csv_path = str(Path(video_path).with_name(f"{Path(video_path).stem}_abertura_rotated.csv"))

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #panel_width = frame_width // 2
    #panel_height = frame_height
    panel_width = frame_width // 2 * 2  # Double the width for better visibility
    panel_height = frame_height * 2 # Double the height for better visibility
    output_width = panel_width * 2
    output_height = panel_height * 3

    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    overlay_writer = cv2.VideoWriter(overlay_output_path, fourcc, fps, (panel_width * 2, panel_height))

    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'left_opening', 'right_opening', 'left_opening_after_rotation', 'right_opening_after_rotation'])

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mid = frame_rgb.shape[1] // 2
        left_eye = frame_rgb[:, :mid]
        right_eye = frame_rgb[:, mid:]

        top_left = make_panel(left_eye, 'Left eye', (panel_width, panel_height))
        top_right = make_panel(right_eye, 'Right eye', (panel_width, panel_height))

        left_edges = preprocess_frame(left_eye)
        right_edges = preprocess_frame(right_eye)

        left_iris_mask, left_eye_iris_bottom, left_eye_iris_top, \
            _, _, left_img_gray_upscaled, _, _, _, left_iris_mask_morph_open = get_iris_mask(left_eye)
        right_iris_mask, right_eye_iris_bottom, right_eye_iris_top, \
            _, _, right_img_gray_upscaled, _, _, _, right_iris_mask_morph_open = get_iris_mask(right_eye)
        
        # Calculate the angles of the iris masks
        left_angle = get_angle_from_mask(left_iris_mask)
        right_angle = get_angle_from_mask(right_iris_mask)

        # Scale up the eye original images 
        upscale_factor = 2.0
        left_img_upscaled = cv2.resize(left_eye, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
        right_img_upscaled = cv2.resize(right_eye, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

        # Rotate the images and masks based on the calculated angles
        left_eye_rotated, left_iris_mask_rotated = rotate_image_and_mask(
            left_img_upscaled, left_iris_mask_morph_open, -left_angle
        )
        right_eye_rotated, right_iris_mask_rotated = rotate_image_and_mask(
            right_img_upscaled, right_iris_mask_morph_open, -right_angle
        )

        # Get the new top and bottom coordinates of the iris masks after rotation
        left_eye_iris_top_after_rotation, left_eye_iris_bottom_after_rotation = get_top_bottom_opening_from_mask(left_iris_mask_rotated)
        right_eye_iris_top_after_rotation, right_eye_iris_bottom_after_rotation = get_top_bottom_opening_from_mask(right_iris_mask_rotated)
        left_opening_after_rotation = max(0.0, float(left_eye_iris_bottom_after_rotation) - float(left_eye_iris_top_after_rotation))
        right_opening_after_rotation = max(0.0, float(right_eye_iris_bottom_after_rotation) - float(right_eye_iris_top_after_rotation))

        left_eye_iris_top, left_eye_iris_bottom = get_top_bottom_opening_from_mask(left_iris_mask)
        right_eye_iris_top, right_eye_iris_bottom = get_top_bottom_opening_from_mask(right_iris_mask)
        left_opening = max(0.0, float(left_eye_iris_bottom) - float(left_eye_iris_top))
        right_opening = max(0.0, float(right_eye_iris_bottom) - float(right_eye_iris_top))

        csv_writer.writerow([frame_count, f'{left_opening:.2f}', f'{right_opening:.2f}', 
                             f'{left_opening_after_rotation:.2f}', f'{right_opening_after_rotation:.2f}'])

        # print(f"Frame {frame_count}: Left iris top={left_eye_iris_top:.1f}, bottom={left_eye_iris_bottom:.1f}, Right iris top={right_eye_iris_top:.1f}, bottom={right_eye_iris_bottom:.1f}")

        middle_left = make_panel(left_edges, 'Left edges', (panel_width, panel_height), is_gray=True)
        middle_right = make_panel(right_edges, 'Right edges', (panel_width, panel_height), is_gray=True)

        bottom_left = make_panel(left_iris_mask, 'Left iris mask', (panel_width, panel_height), is_gray=True)
        bottom_right = make_panel(right_iris_mask, 'Right iris mask', (panel_width, panel_height), is_gray=True)

        bottom_left = annotate_iris_panel(bottom_left, left_iris_mask, left_eye_iris_top, left_eye_iris_bottom)
        bottom_right = annotate_iris_panel(bottom_right, right_iris_mask, right_eye_iris_top, right_eye_iris_bottom)

        top_row = np.hstack((top_left, top_right))
        middle_row = np.hstack((middle_left, middle_right))
        bottom_row = np.hstack((bottom_left, bottom_right))
        montage = np.vstack((top_row, middle_row, bottom_row))

        overlay_left = overlay_iris_mask(left_eye_rotated, left_iris_mask_rotated, left_eye_iris_top_after_rotation, 
                                         left_eye_iris_bottom_after_rotation, 'Left')
        overlay_right = overlay_iris_mask(right_eye_rotated, right_iris_mask_rotated, right_eye_iris_top_after_rotation, 
                                          right_eye_iris_bottom_after_rotation, 'Right')

        #print(f"Overlay left shape: {overlay_left.shape}, overlay right shape: {overlay_right.shape}")
        #print(f"Panel Size: {panel_width}x{panel_height}")

        overlay_frame = np.zeros((panel_height, panel_width * 2, 3), dtype=np.uint8)
        #print(f"Overlay frame shape: {overlay_frame.shape}, expected shape: ({panel_height}, {panel_width * 2}, 3)")
        overlay_frame[:, :panel_width] = overlay_left
        overlay_frame[:, panel_width:] = overlay_right

        writer.write(montage)
        overlay_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
        frame_count += 1

    cap.release()
    writer.release()
    overlay_writer.release()
    csv_file.close()

    print(f"Processed {frame_count} frames.")
    print(f"Output video saved to: {output_path}")
    print(f"Overlay video saved to: {overlay_output_path}")
    print(f"Opening CSV saved to: {csv_path}")
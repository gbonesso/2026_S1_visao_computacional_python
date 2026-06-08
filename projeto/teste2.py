import cv2
import csv
import numpy as np
from pathlib import Path

from pre_processamento import get_iris_mask, preprocess_frame

video_path = './projeto/videos/1_Isabelle_RT_Pre_2025-04-02T13_01_53_362128_output.mp4'
output_path = './projeto/videos/1_Isabelle_RT_Pre_2025-04-02T13_01_53_362128_output_montage.mp4'
overlay_output_path = './projeto/videos/1_Isabelle_RT_Pre_2025-04-02T13_01_53_362128_output_iris_overlay.mp4'
csv_path = str(Path(video_path).with_name(f"{Path(video_path).stem}_abertura.csv"))

# Open the video file
cap = cv2.VideoCapture(video_path)

def make_panel(image, label, size, is_gray=False):
    if is_gray:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    panel = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
    cv2.putText(
        panel,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return panel

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
    panel_width = frame_width // 2
    panel_height = frame_height
    output_width = panel_width * 2
    output_height = panel_height * 3

    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    overlay_writer = cv2.VideoWriter(overlay_output_path, fourcc, fps, (frame_width, frame_height))

    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'left_opening', 'right_opening'])

    

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
                (10, 50), # Posição (x, y) do texto
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, # Tamanho da fonte
                (0, 255, 255),
                1, # Espessura da linha do texto
                cv2.LINE_AA,
            )

        return panel

    def overlay_iris_mask(eye_rgb, iris_mask, iris_top, iris_bottom, eye_label, color=(255, 0, 0), alpha=0.35):
        overlay = eye_rgb.copy()
        if iris_mask.size == 0:
            return overlay

        source_mask_height = iris_mask.shape[0]

        if iris_mask.shape[:2] != eye_rgb.shape[:2]:
            iris_mask = cv2.resize(
                iris_mask,
                (eye_rgb.shape[1], eye_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask = iris_mask > 0
        if not np.any(mask):
            return overlay

        color_layer = np.zeros_like(eye_rgb)
        color_layer[:, :] = color
        overlay[mask] = (
            eye_rgb[mask].astype(np.float32) * (1.0 - alpha)
            + color_layer[mask].astype(np.float32) * alpha
        ).astype(np.uint8)

        top_y = 0.0
        bottom_y = 0.0
        if source_mask_height > 0:
            scale_y = float(eye_rgb.shape[0]) / float(source_mask_height)
            top_y = float(iris_top) * scale_y
            bottom_y = float(iris_bottom) * scale_y

        draw_dotted_hline(overlay, top_y, (0, 255, 255))
        draw_dotted_hline(overlay, bottom_y, (255, 0, 255))

        opening = max(0.0, float(iris_bottom) - float(iris_top))
        cv2.putText(
            overlay,
            f'{eye_label}: {opening:.1f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return overlay

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

        left_iris_mask, left_eye_iris_bottom, left_eye_iris_top = get_iris_mask(left_eye)
        right_iris_mask, right_eye_iris_bottom, right_eye_iris_top = get_iris_mask(right_eye)

        left_opening = max(0.0, float(left_eye_iris_bottom) - float(left_eye_iris_top))
        right_opening = max(0.0, float(right_eye_iris_bottom) - float(right_eye_iris_top))

        csv_writer.writerow([frame_count, f'{left_opening:.2f}', f'{right_opening:.2f}'])

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

        overlay_left = overlay_iris_mask(left_eye, left_iris_mask, left_eye_iris_top, left_eye_iris_bottom, 'Left')
        overlay_right = overlay_iris_mask(right_eye, right_iris_mask, right_eye_iris_top, right_eye_iris_bottom, 'Right')

        overlay_frame = frame_rgb.copy()
        overlay_frame[:, :mid] = overlay_left
        overlay_frame[:, mid:] = overlay_right

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
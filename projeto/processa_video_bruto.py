import cv2
import csv
import numpy as np
from pathlib import Path
from collections import deque

from pre_processamento import get_iris_mask, preprocess_frame
from processa_video_bruto_passo1 import annotate_iris_panel, make_panel, overlay_iris_mask
from pos_processamento import get_angle_from_mask, get_top_bottom_opening_from_mask, rotate_image_and_mask

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}


def process_video_bruto(
    video_input_path,
    output_dir,
    generate_montage_video=False,
    apply_rotation_correction=False,
    rotation_average_window=5,
):
    video_input_path = Path(video_input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_prefix = video_input_path.stem.split('_', 1)[0]

    if rotation_average_window <= 0:
        raise ValueError("rotation_average_window deve ser maior que zero.")

    output_path = output_dir / f"{video_input_path.stem}_montage.mp4"
    overlay_output_path = output_dir / f"{video_input_path.stem}_overlay.mp4"
    csv_path = output_dir / f"{video_prefix}_output_abertura.csv"

    cap = cv2.VideoCapture(str(video_input_path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    panel_width = frame_width // 2 * 2
    panel_height = frame_height * 2
    output_width = panel_width * 2
    output_height = panel_height * 3

    writer = None
    if generate_montage_video:
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    overlay_writer = cv2.VideoWriter(str(overlay_output_path), fourcc, fps, (panel_width * 2, panel_height))

    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'left_opening', 'right_opening', 'left_opening_after_rotation', 'right_opening_after_rotation'])

    frame_count = 0
    left_angle_history = deque(maxlen=rotation_average_window)
    right_angle_history = deque(maxlen=rotation_average_window)

    try:
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

            left_angle = get_angle_from_mask(left_iris_mask)
            right_angle = get_angle_from_mask(right_iris_mask)

            left_angle_history.append(left_angle)
            right_angle_history.append(right_angle)

            left_angle_smoothed = float(np.mean(left_angle_history))
            right_angle_smoothed = float(np.mean(right_angle_history))

            upscale_factor = 2.0
            left_img_upscaled = cv2.resize(left_eye, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
            right_img_upscaled = cv2.resize(right_eye, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

            left_eye_iris_top, left_eye_iris_bottom = get_top_bottom_opening_from_mask(left_iris_mask)
            right_eye_iris_top, right_eye_iris_bottom = get_top_bottom_opening_from_mask(right_iris_mask)
            left_opening = max(0.0, float(left_eye_iris_bottom) - float(left_eye_iris_top))
            right_opening = max(0.0, float(right_eye_iris_bottom) - float(right_eye_iris_top))

            if apply_rotation_correction:
                left_eye_rotated, left_iris_mask_rotated = rotate_image_and_mask(
                    left_img_upscaled, left_iris_mask_morph_open, -left_angle_smoothed
                )
                right_eye_rotated, right_iris_mask_rotated = rotate_image_and_mask(
                    right_img_upscaled, right_iris_mask_morph_open, -right_angle_smoothed
                )
            else:
                left_eye_rotated, left_iris_mask_rotated = left_img_upscaled, left_iris_mask_morph_open
                right_eye_rotated, right_iris_mask_rotated = right_img_upscaled, right_iris_mask_morph_open

            if apply_rotation_correction:
                left_eye_iris_top_after_rotation, left_eye_iris_bottom_after_rotation = get_top_bottom_opening_from_mask(left_iris_mask_rotated)
                right_eye_iris_top_after_rotation, right_eye_iris_bottom_after_rotation = get_top_bottom_opening_from_mask(right_iris_mask_rotated)
                left_opening_after_rotation = max(0.0, float(left_eye_iris_bottom_after_rotation) - float(left_eye_iris_top_after_rotation))
                right_opening_after_rotation = max(0.0, float(right_eye_iris_bottom_after_rotation) - float(right_eye_iris_top_after_rotation))
            else:
                left_eye_iris_top_after_rotation, left_eye_iris_bottom_after_rotation = left_eye_iris_top, left_eye_iris_bottom
                right_eye_iris_top_after_rotation, right_eye_iris_bottom_after_rotation = right_eye_iris_top, right_eye_iris_bottom
                left_opening_after_rotation = left_opening
                right_opening_after_rotation = right_opening

            csv_writer.writerow([frame_count, f'{left_opening:.2f}', f'{right_opening:.2f}',
                                 f'{left_opening_after_rotation:.2f}', f'{right_opening_after_rotation:.2f}'])

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

            overlay_frame = np.zeros((panel_height, panel_width * 2, 3), dtype=np.uint8)
            overlay_frame[:, :panel_width] = overlay_left
            overlay_frame[:, panel_width:] = overlay_right

            if generate_montage_video and writer is not None:
                writer.write(montage)
            overlay_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
            frame_count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        overlay_writer.release()
        csv_file.close()

    print(f"Processed {frame_count} frames.")
    if generate_montage_video:
        print(f"Output video saved to: {output_path}")
    print(f"Overlay video saved to: {overlay_output_path}")
    print(f"Opening CSV saved to: {csv_path}")


def process_videos_in_directory(
    input_dir,
    output_dir,
    generate_montage_video=False,
    apply_rotation_correction=False,
    rotation_average_window=5,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Diretório de entrada inválido: {input_dir}")

    video_files = sorted(
        video_path
        for video_path in input_dir.iterdir()
        if video_path.is_file()
        and video_path.suffix.lower() in VIDEO_EXTENSIONS
        and not video_path.stem.endswith('_montage')
        and not video_path.stem.endswith('_overlay')
    )

    if not video_files:
        print(f"Nenhum vídeo encontrado em: {input_dir}")
        return

    for video_path in video_files:
        print(f"Processando: {video_path.name}")
        process_video_bruto(
            video_path,
            output_dir,
            generate_montage_video=generate_montage_video,
            apply_rotation_correction=apply_rotation_correction,
            rotation_average_window=rotation_average_window,
        )


if __name__ == '__main__':
    process_videos_in_directory(
        './projeto/videos_input',
        './projeto/videos_output_com_rotacao_com_med_ang_5',
        generate_montage_video=False,
        apply_rotation_correction=True,
        rotation_average_window=5,
    )
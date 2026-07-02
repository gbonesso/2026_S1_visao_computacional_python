from __future__ import annotations

import cv2
import numpy as np


def get_angle_from_mask(mask: np.ndarray) -> float:
    """Return the contour alignment angle, in degrees, for a binary mask."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("A máscara não contém contornos.")

    contour = max(contours, key=cv2.contourArea)

    vx, vy, x0, y0 = cv2.fitLine(
        contour,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    )

    vx = float(np.squeeze(vx).item())
    vy = float(np.squeeze(vy).item())

    angle = np.degrees(np.arctan2(vy, vx))
    return float(angle)

def rotate_image_and_mask(image, mask, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    rotated_image = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    rotated_mask = cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_image, rotated_mask


def get_top_bottom_opening_from_mask(
    mask: np.ndarray,
    top_percentile: float = 5,
    bottom_percentile: float = 95,
) -> tuple[float, float]:
    """Return the opening, in pixels, measured from mask y-coordinates percentiles."""
    y_coords = np.where(mask > 0)[0]

    if y_coords.size == 0:
        raise ValueError("A máscara não contém pixels válidos.")

    iris_bottom = np.percentile(y_coords, bottom_percentile)
    iris_top = np.percentile(y_coords, top_percentile)

    return float(iris_top), float(iris_bottom)
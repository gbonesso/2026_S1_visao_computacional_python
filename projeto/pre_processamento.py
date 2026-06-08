import cv2
import numpy as np

"""
This module contains the preprocessing steps for eye frames, including:
- Grayscale conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian Blur
- Upscaling
- Canny Edge Detection
- ROI Masking
- Connected Components Filtering
- Morphological Closing"""


def keep_largest_component(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

    if num_labels <= 1:
        return binary_mask

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    filtered = np.zeros_like(binary_mask)
    filtered[labels == largest_label] = 255
    return filtered


def preprocess_frame(eye_frame):
    # Create img_gray from eye_frame
    img_gray = cv2.cvtColor(eye_frame, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)

    # Use GaussianBlur as in EyeOpenEstimator and apply before upscaling
    img_gray_blurred = cv2.GaussianBlur(img_gray, (5,5), 0)

    # Upscale img_gray_blurred
    upscale_factor = 2
    img_gray_upscaled = cv2.resize(
        img_gray_blurred, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR
    )

    # Apply Canny Edge Detector with fixed thresholds (as in EyeOpenEstimator)
    edges = cv2.Canny(img_gray_upscaled, 30, 100)

    # Apply ROI mask (as in EyeOpenEstimator)
    h, w = edges.shape
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    y0 = int(h * 0.25)
    y1 = int(h * 0.75)
    mask = np.zeros_like(edges)
    mask[y0:y1, x0:x1] = 255
    edges = cv2.bitwise_and(edges, mask)

    # Filtrar componentes conectados (adjusted area threshold as in EyeOpenEstimator)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)

    # Create a color map for visualization of connected components
    colored_labels = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

    filtered = np.zeros_like(edges)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 40: # Using 40 as in EyeOpenEstimator
            filtered[labels == i] = 255
            # Assign a random color to each connected component for visualization
            color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
            colored_labels[labels == i] = color

    edges = filtered # Update edges to filtered for curve extraction

    # Add morphological closing to connect fragmented edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges

def get_iris_mask(eye_frame):
    # Create img_gray from eye_frame
    img_gray = cv2.cvtColor(eye_frame, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)

    # Use GaussianBlur as in EyeOpenEstimator and apply before upscaling
    img_gray_blurred = cv2.GaussianBlur(img_clahe, (5,5), 0)

    # Upscale img_gray_blurred
    upscale_factor = 2
    img_gray_upscaled = cv2.resize(
        img_gray_blurred, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR
    )

    # Define ROI mask so pixels outside the band stay black in the final mask
    h, w = img_gray_upscaled.shape
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    y0 = int(h * 0.25)
    y1 = int(h * 0.75)
    mask = np.zeros_like(img_gray_upscaled)
    mask[y0:y1, x0:x1] = 255

    # --- Pupil/Iris Detection and new Opening Calculation ---
    opening = 0 # Default value
    iris_bottom = 0 # Initialize iris_bottom
    iris_top = 0 # Initialize iris_top

    # Detect dark region (iris/pupil) on the upscaled grayscale image
    _, iris_mask = cv2.threshold(
        img_gray_upscaled, 
        55, # This threshold can be adjusted based on the lighting conditions and eye color
        255, # Max value for binary thresholding
        cv2.THRESH_BINARY_INV
    )
    iris_mask = cv2.bitwise_and(iris_mask, mask)

    # Expand the final selected region slightly to increase detected area
    dil_kernel = np.ones((5, 5), np.uint8)
    iris_mask = cv2.dilate(iris_mask, dil_kernel, iterations=2)

    # Light morphology: connect nearby fragments without exploding the region
    kernel = np.ones((9, 9), np.uint8)
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    iris_mask = keep_largest_component(iris_mask)

    # Remove any small disconnected leftovers after the largest component was selected.
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    iris_mask = keep_largest_component(iris_mask)

    # Find the y-coordinates of the iris pixels
    iris_y_coords = np.where(iris_mask > 0)[0]

    if len(iris_y_coords) > 0:
        # Get the 90th percentile as the bottom edge of the iris
        iris_bottom = np.percentile(iris_y_coords, 95)
        # Get the 10th percentile as the top edge of the iris
        iris_top = np.percentile(iris_y_coords, 5)

    return iris_mask, iris_bottom, iris_top, img_clahe

def get_iris_mask_v2(eye_frame, use_upscale=False, upscale_factor=2, dilate_iterations=1):
    # Create img_gray from eye_frame
    img_gray = cv2.cvtColor(eye_frame, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)

    # Use GaussianBlur before optional upscaling
    img_gray_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    if use_upscale:
        img_gray_processed = cv2.resize(
            img_gray_blurred,
            None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        img_gray_processed = img_gray_blurred

    h, w = img_gray_processed.shape
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    y0 = int(h * 0.22)
    y1 = int(h * 0.78)

    roi_mask = np.zeros_like(img_gray_processed)
    roi_mask[y0:y1, x0:x1] = 255

    roi_band = img_gray_processed[y0:y1, x0:x1]

    # Adaptive threshold on the ROI band keeps small iris fragments visible.
    band_mask = cv2.adaptiveThreshold(
        roi_band,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # This method computes the threshold for a pixel based on a small neighborhood around it, which helps in varying lighting conditions.
        cv2.THRESH_BINARY_INV,
        41, # Block size for adaptive thresholding (must be odd and >1)
        1,  # Constant subtracted from the mean (adjustable based on lighting conditions
    )

    # Rebuild a full-size mask with black outside the ROI band
    iris_mask = np.zeros_like(img_gray_processed)
    iris_mask[y0:y1, x0:x1] = band_mask

    # Light morphology: connect nearby fragments without exploding the region
    kernel = np.ones((3, 3), np.uint8)
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    iris_mask = keep_largest_component(iris_mask)

    # Keep components whose area fits a plausible iris/pupil band size
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(iris_mask)
    filtered = np.zeros_like(iris_mask)

    roi_area = float((y1 - y0) * w)
    min_area = max(10, int(roi_area * 0.0005))
    max_area = int(roi_area * 0.55)

    center_x = w / 2.0
    kept_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        cx = centroids[i][0]
        cy = centroids[i][1]
        center_penalty = abs(cx - center_x) / max(1.0, w)
        vertical_penalty = abs(cy - (y0 + y1) / 2.0) / max(1.0, h)
        if center_penalty <= 0.35 and vertical_penalty <= 0.28:
            filtered[labels == i] = 255
            kept_area += area

    if kept_area == 0 and num_labels > 1:
        # Fallback: keep the largest valid component when the strict filter finds nothing.
        fallback_label = -1
        fallback_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            if area > fallback_area:
                fallback_area = area
                fallback_label = i

        if fallback_label > 0:
            filtered[labels == fallback_label] = 255

    if cv2.countNonZero(filtered) > 0:
        # Fill holes inside the selected region and recover tiny gaps.
        flood_filled = filtered.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
        flood_filled_inv = cv2.bitwise_not(flood_filled)
        filtered = cv2.bitwise_or(filtered, flood_filled_inv)

        if cv2.countNonZero(filtered) < int(roi_area * 0.01):
            filtered = cv2.dilate(filtered, kernel, iterations=1)

    filtered = keep_largest_component(filtered)

    # Optionally expand the final selected region slightly to increase detected area
    if dilate_iterations and cv2.countNonZero(filtered) > 0:
        dil_kernel = np.ones((3, 3), np.uint8)
        iris_mask = cv2.dilate(filtered, dil_kernel, iterations=dilate_iterations)
    else:
        iris_mask = filtered

    opening = 0
    iris_bottom = 0
    iris_top = 0

    iris_y_coords = np.where(iris_mask > 0)[0]

    if len(iris_y_coords) > 0:
        iris_bottom = np.percentile(iris_y_coords, 90)
        iris_top = np.percentile(iris_y_coords, 10)

    return iris_mask, iris_bottom, iris_top
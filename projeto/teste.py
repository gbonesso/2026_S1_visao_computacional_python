import cv2
import matplotlib.pyplot as plt
import numpy as np

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

# Separar os olhos
mid = frame_rgb.shape[1] // 2
left_eye = frame_rgb[:, :mid]
right_eye = frame_rgb[:, mid:]

# Create img_gray from left_eye
img_gray = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_gray = clahe.apply(img_gray)

# Use GaussianBlur as in EyeOpenEstimator and apply before upscaling
img_gray_blurred = cv2.GaussianBlur(img_gray, (5,5), 0)

# Upscale img_gray_blurred
upscale_factor = 2
img_gray_upscaled = cv2.resize(img_gray_blurred, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

# Histogram of the upscaled grayscale image
plt.figure(figsize=(10, 4))
plt.title("Histogram of img_gray_upscaled")
plt.hist(img_gray_upscaled.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.85)
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Apply Canny Edge Detector with fixed thresholds (as in EyeOpenEstimator)
edges = cv2.Canny(img_gray_upscaled, 30, 100)

# Apply ROI mask (as in EyeOpenEstimator)
h, w = edges.shape
mask = np.zeros_like(edges)
mask[int(h*0.25):int(h*0.75), :] = 255
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

# Detectar a pálpebra superior e inferior
top_curve = []
bottom_curve = []

for col in range(edges.shape[1]):
    ys = np.where(edges[:, col] > 0)[0]
    if len(ys) > 0:
        top_curve.append(ys[0])
        bottom_curve.append(ys[-1]) # Get the last (bottommost) edge

# Converter em sinal util
top_curve = np.array(top_curve)
bottom_curve = np.array(bottom_curve)

# --- Pupil/Iris Detection and new Opening Calculation ---
opening = 0 # Default value
iris_bottom = 0 # Initialize iris_bottom



# Detect dark region (iris/pupil) on the upscaled grayscale image
_, iris_mask = cv2.threshold(img_gray_upscaled, 50, 255, cv2.THRESH_BINARY_INV)

# Find the y-coordinates of the iris pixels
iris_y_coords = np.where(iris_mask > 0)[0]

if len(iris_y_coords) > 0:
    # Get the 90th percentile as the bottom edge of the iris
    iris_bottom = np.percentile(iris_y_coords, 90)

# Calculate opening with the new strategy: iris_bottom - median(top_curve)
if len(top_curve) > 0 and iris_bottom > 0:
    # Calculate opening as the distance between the iris bottom and the median of the top eyelid curve
    opening = iris_bottom - np.median(top_curve)
else:
    opening = 0 # Default if no sufficient data for calculation

print("opening: ", opening)
# --- End of new Opening Calculation ---

# Visualize the original grayscale image, Canny edges, and Iris Mask
plt.figure(figsize=(18, 6)) # Increase figure size for 3 subplots

plt.subplot(1, 3, 1) # First subplot
plt.title("Grayscale Image (upscaled and processed)")
plt.imshow(img_gray_upscaled, cmap="gray")
plt.axis('off')

plt.subplot(1, 3, 2) # Second subplot
plt.title("Edges with Eyelid Curves & Iris Bottom")
plt.imshow(edges, cmap="gray")
plt.axis('off')
if len(top_curve) > 0:
    plt.plot(np.arange(len(top_curve)), top_curve, color='red', linewidth=1, label='Top Eyelid')
if len(bottom_curve) > 0:
    plt.plot(np.arange(len(bottom_curve)), bottom_curve, color='blue', linewidth=1, label='Bottom Eyelid')
if iris_bottom > 0: # Only plot if iris_bottom was successfully detected
    plt.axhline(y=iris_bottom, color='yellow', linestyle='--', label='Iris Bottom (90th percentile)')
plt.legend()


plt.subplot(1, 3, 3) # Third subplot for Iris Mask
plt.title("Detected Iris Mask")
plt.imshow(iris_mask, cmap="gray")
plt.axis('off')

plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()
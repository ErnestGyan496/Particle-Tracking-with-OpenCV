import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/Removed_Wing_Images/Frame0000.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the image for inspection
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="gray")
plt.title("Particle Image")
plt.axis("off")
plt.show()

# Shi-Tomasi corner detection parameters
feature_params = dict(
    maxCorners=300,  # Allow up to 300 corners
    qualityLevel=0.2,  # Lower quality threshold for more sensitivity
    minDistance=5,  # Minimum distance between detected corners
    blockSize=7,  # Block size for corner detection
)

# Detect corners
corners = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)

# Create a copy of the original image for visualization
image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw detected corners as small circles
if corners is not None:
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_with_corners, (int(x), int(y)), 3, (0, 255, 0), -1)

# Display the result with detected corners
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
plt.title("Detected Particles (Shi-Tomasi Corners)")
plt.axis("off")
plt.show()

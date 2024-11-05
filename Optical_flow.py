import numpy as np
import cv2
import glob
import os

# Load Images
output_folder = r"/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/Removed_Wing_Images"
image_paths = sorted(glob.glob(os.path.join(output_folder, "*.tif")))
output_dir = "/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/Optical_Images"

if not image_paths:
    print("No .tif images found in the specified folder.")
    exit()

# Load the first image as the starting frame
first_frame = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
if first_frame is None:
    print("Image could not be loaded. Please check the file path and format.")
    exit()

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=5, blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Find corners in the first frame
p0 = cv2.goodFeaturesToTrack(first_frame, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(cv2.imread(image_paths[0]))

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Iterate over each consecutive image to simulate optical flow
for i in range(260, len(image_paths)):
    frame = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print(f"Frame {image_paths[i]} could not be loaded.")
        continue

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(first_frame, frame, p0, None, **lk_params)

    # Select good points (those with status `st == 1`)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Draw the tracks
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
        frame_colored = cv2.circle(
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
            (int(a), int(b)),
            5,
            color[j].tolist(),
            -1,
        )

    # Combine frame with mask to show tracked paths
    img_with_tracks = cv2.add(frame_colored, mask)

    # Display the image
    cv2.imshow("Optical Flow Tracking", img_with_tracks)

    # Save each output frame
    output_file_path = os.path.join(output_dir, f"tracked_frame_{i}.tif")
    cv2.imwrite(output_file_path, img_with_tracks)
    print(f"Saved tracked frame to {output_file_path}")

    # Update previous frame and points for next iteration
    first_frame = frame.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Break loop if 'Esc' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()

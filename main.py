__My_name__ = "Ernest Bediako"
__date_started__ = "2/11/2024"
__task__ = "Contour Detection"

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def Load_Images_(images_folder, output_folder):

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    renamed_images1 = []

    # Loop through the images in the source folder
    for image in os.listdir(images_folder):
        if image.endswith(".tif"):
            image_path = os.path.join(images_folder, image)
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if "fr" in image[-10:]:
                new_image_name = image[-10:].replace("fr", "Frame")

                # Lets Define the path for saving the new image in the destination folder
                new_image_path = os.path.join(output_folder, new_image_name)

                # Save the image with the new name in the output folder
                cv2.imwrite(new_image_path, img)
                print(f"Saved {new_image_name} to {output_folder}")
                renamed_images1.append(new_image_path)
            else:
                print(f"No 'fr' found in the expected position in file {image}")
        else:
            print(f" {image} is not a tif file")
    return renamed_images1


def contour_technique1(image_path):
    image_path = glob.glob(os.path.join(output_folder, "*.tif"))
    img = cv2.imread(image_path[0], cv2.IMREAD_UNCHANGED)

    cv2.imshow("Original_Image", img)

    # blur_image = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(img, 60, 250, cv2.THRESH_BINARY)
    thresh = cv2.convertScaleAbs(thresh)

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour_image = cv2.drawContours(img, contours, -1, (60, 250, 0), 3)
    cnt = contours[0]
    M = cv2.moments(cnt)
    print(M)

    if M["m00"] != 0:
        x_coordinate = int(M["m10"] / M["m00"])
        y_coordinate = int(M["m01"] / M["m00"])
        # the centroid
        print("Centroid", x_coordinate, y_coordinate)
        cv2.circle(input_image, (x_coordinate, y_coordinate), 1, (0, 0, 250), -1)
        cv2.imshow("Centroid", contour_image)
    else:
        print("Contour area is zero; centroid cannot be computed.")

    cv2.waitKey()
    cv2.destroyAllWindows()


def Moment_Calculation(output_folder):
    # Load the first .tif image in the folder
    image_paths = glob.glob(os.path.join(output_folder, "*.tif"))
    if not image_paths:
        print("No .tif images found in the specified folder.")
        return

    img_path = image_paths[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Image could not be loaded. Please check the file path and format.")
        return

    cv2.imshow("Original_Image", img)

    # Apply binary thresholding to separate particles
    _, thresh = cv2.threshold(img, 60, 250, cv2.THRESH_BINARY)
    thresh = cv2.convertScaleAbs(thresh)

    cv2.imshow("Thresholded Image", thresh)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours on a copy of the original image
    contour_image = img.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 2)
    cv2.imshow("Contours", contour_image)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:  # Adjust minimum area threshold based on particle size

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                x_coordinate = int(M["m10"] / M["m00"])
                y_coordinate = int(M["m01"] / M["m00"])
                print("Centroid:", x_coordinate, y_coordinate)

                # Mark the centroid on the image
                cv2.circle(
                    contour_image, (x_coordinate, y_coordinate), 1, (0, 0, 255), -1
                )
            else:
                print("Contour area is zero; centroid cannot be computed.")
        else:
            print("Skipping small contour with area:", area)

    contour = contours[1]
    contour_area = cv2.contourArea(contour)
    print(f"contour area {contour_area}")

    # Display the image with centroids marked
    cv2.imshow("Centroids", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def MaxContour_Area(output_folder):
    # Load the first .tif image in the folder
    image_paths = glob.glob(os.path.join(output_folder, "*.tif"))
    if not image_paths:
        print("No .tif images found in the specified folder.")
        return

    img_path = image_paths[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Image could not be loaded. Please check the file path and format.")
        return

    # Apply binary thresholding to separate particles
    _, thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    thresh = cv2.convertScaleAbs(thresh)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    print(f"Largest contour area: {largest_area}")

    # Create a black mask and draw the largest contour in white
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(
        mask,
        [largest_contour],
        -1,
        (255, 255, 255),
        thickness=cv2.FILLED,
    )

    mask_inverted = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=mask_inverted)

    _, thresh2 = cv2.threshold(result, 60, 255, cv2.THRESH_BINARY)

    # Perform contour detection on the result image
    remaining_contours, _ = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Optional: Create a new mask to visualize the remaining contours
    remaining_contours_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(
        remaining_contours_mask, remaining_contours, -1, (255), thickness=cv2.FILLED
    )

    # Save the images to the output folder
    # cv2.imwrite("Original_Image.png", img)
    # cv2.imwrite("Binary Image.png", thresh)
    # cv2.imwrite("Max_area Mask_with_Largest_Contour.png", mask_with_largest_contour)
    # cv2.imwrite("Max_area After Bitwise Operation.png", result)
    # cv2.imwrite("Max_Area_Final.png", thresh2)

    cv2.imshow("Original Image", img)
    cv2.imshow("Binary Image", thresh)

    cv2.imshow("Mask with Largest Contour", mask)
    cv2.imshow("Max_area After Bitwise Operation", result)
    cv2.imshow("Max_Area_Final", thresh2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ArcLength(output_folder):
    # Load the first .tif image in the folder
    image_paths = glob.glob(os.path.join(output_folder, "*.tif"))
    if not image_paths:
        print("No .tif images found in the specified folder.")
        return

    img_path = image_paths[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Image could not be loaded. Please check the file path and format.")
        return
    _, threshold = cv2.threshold(img, 60, 240, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    longest_contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, True))

    # Create a mask the same size as the original image, initialized to all zeros (black)
    mask = np.zeros_like(img, dtype=np.uint8) * 255

    # Fill the area under the longest contour in the mask with white (255)
    cv2.drawContours(mask, [longest_contour], -1, (255), thickness=cv2.FILLED)
    inverted_mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, inverted_mask)

    _, thresh3 = cv2.threshold(result, 60, 255, cv2.THRESH_BINARY)

    # Save the images to the output folder
    # cv2.imwrite("ArcLen_Original_Image.png", img)
    # cv2.imwrite("ArcLen_Binary.png", threshold)
    # cv2.imwrite("ArcLen_Inverted_Mask.png", inverted_mask)
    # cv2.imwrite("result after bitwise.png", result)
    # cv2.imwrite("ArcLen thresh3 after enhancement.png", thresh3)

    cv2.imshow("Original image", img)
    cv2.imshow("Binary Image", threshold)
    cv2.imshow("Inverted_Mask", inverted_mask)
    cv2.imshow("result after bitwise", result)
    cv2.imshow("thresh3 after enhancement", thresh3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ContourApproximation(output_folder):
    # Load the first .tif image in the folder
    image_paths = glob.glob(os.path.join(output_folder, "*.tif"))
    if not image_paths:
        print("No .tif images found in the specified folder.")
        return

    img_path = image_paths[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Image could not be loaded. Please check the file path and format.")
        return
    _, threshold = cv2.threshold(img, 60, 240, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img, dtype=np.uint8)

    for cnt_ in contours:
        epsilon = 0.001 * cv2.arcLength(cnt_, True)
        approx = cv2.approxPolyDP(cnt_, epsilon, True)
        print("epsilons", epsilon)

        # cv2.drawContours(img, [approx], 0, (240), 3)
        cv2.drawContours(mask, [approx], 0, (240, 0, 0), 2)

        # inverted_mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(img, mask)
        _, thresh3 = cv2.threshold(result, 60, 255, cv2.THRESH_BINARY)

    # Save the images to the output folder
    # cv2.imwrite("Approx_Original_Image.png", img)
    # cv2.imwrite("Approx_threshold.png", threshold)
    # cv2.imwrite("Approx_result.png", result)
    # cv2.imwrite("Approx_Final.png", thresh3)

    cv2.imshow("Approx_Original image", img)
    cv2.imshow("Approx_thresh", threshold)
    cv2.imshow("Approx_mask", mask)
    cv2.imshow("Approx_result", result)
    # cv2.imshow("thresh2", thresh3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ContourApproximation2(output_folder):
    # Load the first .tif image in the folder
    image_paths = glob.glob(os.path.join(output_folder, "*.tif"))
    if not image_paths:
        print("No .tif images found in the specified folder.")
        return

    img_path = image_paths[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Image could not be loaded. Please check the file path and format.")
        return

    # Apply binary thresholding to the image
    _, threshold = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)

    # Find all contours in the binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a mask for only the smaller contours (excluding the two largest)
    mask = np.zeros_like(img, dtype=np.uint8)  # Black mask

    # Draw all contours except the two largest on the mask
    for cnt in contours[2:]:  # Skip the two largest contours
        epsilon = 0.1 * cv2.arcLength(
            cnt, True
        )  # Small epsilon for a tighter approximation
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(mask, [approx], 0, 255, 2)  # Draw smaller contours in white

    # Invert the mask and apply it to the original image
    result = cv2.bitwise_and(img, mask)

    # Additional binary threshold to further separate particles in the result
    _, thresh3 = cv2.threshold(result, 60, 255, cv2.THRESH_BINARY)

    row_index = 100
    intensity_profile = thresh3[row_index, :]

    # # Save the images to the output folder
    # cv2.imwrite("Approx_Original_Image.png", img)
    # cv2.imwrite("Approx_threshold.png", threshold)
    # cv2.imwrite("Approx_mask.png", mask)
    # cv2.imwrite("Approx_result.png", result)
    # cv2.imwrite("Approx_Final.png", thresh3)

    # Display the images (for debugging or visualization purposes)
    cv2.imshow("Approx_Original Image", img)
    cv2.imshow("Approx_thresh", threshold)
    cv2.imshow("Approx_mask with Smaller Contours", mask)
    cv2.imshow("Approx_result with Largest Contours Removed", result)
    cv2.imshow("Approx_Final Thresholded Result", thresh3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def largest_Area2(output_folder):
    # Load the first .tif image in the folder
    image_paths = glob.glob(os.path.join(output_folder, "*.tif"))
    if not image_paths:
        print("No .tif images found in the specified folder.")
        return

    img_path = image_paths[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Image could not be loaded. Please check the file path and format.")
        return

    # Apply binary thresholding to the image
    _, threshold = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)

    # Find all contours in the binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    Largest_areas = sorted(contours, key=cv2.contourArea, reverse=True)
    print(len(Largest_areas))

    # create a mask
    mask = np.zeros_like(img, dtype=np.uint8)  # Black mask

    contour_on_Mask = cv2.drawContours(
        mask, Largest_areas[:], 0, (255, 255, 255), thickness=cv2.FILLED
    )

    # img_bitContour = cv2.bitwise_or(img, img, mask)
    mask_not_img = cv2.bitwise_not(mask)
    mask_merge_ = cv2.bitwise_and(img, img, mask=mask_not_img)

    _, threshold2 = cv2.threshold(mask_merge_, 60, 250, cv2.THRESH_BINARY)

    cv2.imshow("contour_on_Mask", contour_on_Mask)
    cv2.imshow("mask_not_img", mask_not_img)
    cv2.imshow("mask merge", mask_merge_)
    cv2.imshow("enhancement", threshold2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_folder = r"/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/1p95Kell1_00_22-11-23_12-52-34p714"
    output_folder = r"/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/Loaded_Images1"

    # Load_Images_(images_folder, output_folder)
    # contour_technique1(output_folder)

    # Moment_Calculation(output_folder)
    # MaxContour_Area(output_folder)
    # ArcLength(output_folder)
    # ContourApproximation(output_folder)
    # ContourApproximation2(output_folder)
    largest_Area2(output_folder)

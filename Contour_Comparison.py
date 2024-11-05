__My_name__ = "Ernest Bediako"
__date_started__ = "2/11/2024"
__task__ = "Contour Detection"

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def simple_Contour(output_folder):
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
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(img, dtype=np.uint8)

    cv2.drawContours(
        mask,
        contours,
        -1,
        (255, 255, 255),
        thickness=cv2.FILLED,
    )
    cv2.drawContours(
        img,
        contours,
        -1,
        (255, 255, 255),
        6,
    )
    # cv2.imwrite("Chain_approx_SIMPLE_RETR_EXTERNAL.png", img)
    # cv2.imwrite("Chain_approx_SIMPLE_RETR_EXTERNAL_on_Mask.png", mask)

    # cv2.imwrite("Chain_approx_SIMPLE_RETR_LIST.png", img)
    # cv2.imwrite("Chain_approx_SIMPLE_RETR_LIST_on_Mask.png", mask)

    # cv2.imwrite("Chain_approx_SIMPLE_RETR_CCOMP.png", img)
    # cv2.imwrite("Chain_approx_SIMPLE_RETR_CCOMP_on_Mask.png", mask)

    # cv2.imwrite("Chain_approx_SIMPLE_RETR_TREE.png", img)
    # cv2.imwrite("Chain_approx_SIMPLE_RETR_TREE_on_Mask.png", mask)

    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def None_Contour(output_folder):
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
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    mask = np.zeros_like(img, dtype=np.uint8)

    cv2.drawContours(
        img,
        contours,
        -1,
        (255, 255, 255),
        5,
    )
    cv2.drawContours(
        mask,
        contours[:],
        -1,
        (255),
        thickness=cv2.FILLED,
    )

    # _mask = cv2.bitwise_not(mask)
    # result = cv2.bitwise_and(img, _mask)

    # cv2.imwrite("Chain_approx_NONE_RETR_TREE.png", img)
    # cv2.imwrite("Chain_approx_NONE_RETR_TREE_on_Mask.png", mask)

    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    # cv2.imshow("_Mask", _mask)
    # cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def TC89_KCOS_Contour(output_folder):
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
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    mask = np.zeros_like(img, dtype=np.uint8)

    # cv2.drawContours(
    #     img,
    #     contours,
    #     -1,
    #     (255, 255, 255),
    #     6,
    # )
    cv2.drawContours(
        img,
        contours,
        -1,
        (255, 255, 255),
        thickness=cv2.FILLED,
    )

    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_folder = r"/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/1p95Kell1_00_22-11-23_12-52-34p714"
    output_folder = r"/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/Loaded_Images1"

    simple_Contour(output_folder)
    # None_Contour(output_folder)
    # TC89_KCOS_Contour(output_folder)

import os
import cv2
import glob


def Load_Images_(images_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder at: {output_folder}")

    renamed_images1 = []
    count = 0

    # Loop through the images in the source folder
    for image in os.listdir(images_folder):
        if image.endswith(".tif"):
            image_path = os.path.join(images_folder, image)
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(
                    f"Failed to load image {image_path}. It may be corrupted or not a valid image."
                )
                continue

            # Process only if "fr" is found in the last 10 characters and only until 10 images
            if "fr" in image[-10:] and count < 10:
                new_image_name = image[-10:].replace("fr", "Frame")
                new_image_path = os.path.join(output_folder, new_image_name)

                print(f"Attempting to save image at: {new_image_path}")

                # Save the image with the new name in the output folder
                cv2.imwrite(new_image_path, img)
                print(f"Saved {new_image_name} to {output_folder}")
                renamed_images1.append(new_image_path)

                count += 1  # Increment the counter

                # Stop once 10 images have been saved
                if count >= 10:
                    print("Saved 10 images, stopping.")
                    break
            else:
                print(f"No 'fr' found in the expected position in file {image}")
        else:
            print(f"{image} is not a .tif file")

    return renamed_images1


def contour_technique1(image_path):
    image_path = glob.glob(os.path.join(output_folder, "*.tif"))
    img = cv2.imread(image_path[0], cv2.IMREAD_UNCHANGED)

    cv2.imshow("Original_Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_folder = r"/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/1p95Kell1_00_22-11-23_12-52-34p714"
    output_folder = r"/Users/bullet/Desktop/Machine_Learning projects_2024/Image_Processing/contourAnalysis/Loaded_Images1"

    Load_Images_(images_folder, output_folder)
    contour_technique1(output_folder)

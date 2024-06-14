import cv2
from stitching import Stitcher
import numpy as np

def stitch_microscope_images(images):
    stitcher = Stitcher()
    panorama = stitcher.stitch(images)
    return panorama

def count_cell_colonies(panorama):
    # Convert the image to grayscale
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve detection accuracy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to remove small noise and emphasize the features
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size to identify cell colonies
    num_colonies = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Assuming each colony has an area greater than a certain threshold
            num_colonies += 1

    return num_colonies

def main():
    # Assuming microscope images are passed as input arguments
    images = [cv2.imread(image) for image in input("Enter image filepaths separated by space: ").split()]

    # Stitch microscope images together to form a panorama
    panorama = stitch_microscope_images(images)

    # Count cell colonies in the panorama using ORB feature extraction
    num_colonies = count_cell_colonies(panorama)

    print("Number of cell colonies:", num_colonies)

if __name__ == "__main__":
    main()

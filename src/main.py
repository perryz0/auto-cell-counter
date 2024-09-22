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

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(blurred, None)

    # Use Brute Force Matcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Placeholder logic for now, counts colonies based on the num of keypoints
    # ***PENDING LOGIC ADJUSTMENTS AFTER TESTING
    num_colonies = len(keypoints) // 50  # Placeholder counting logic (requirement for something to be considered a colony)

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

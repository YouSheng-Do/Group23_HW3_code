import cv2
import numpy as np
from utils import find_homography
from step1_2 import detect_and_describe, match_features
from step4 import warp

def stitch_images(img1, img2):
    # 1. Interest points detection & feature description by SIFT
    keypoints1, descriptors1 = detect_and_describe(img1, "Image 1")
    keypoints2, descriptors2 = detect_and_describe(img2, "Image 2")
    
    # 2. Feature matching by SIFT features
    matches = match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2)
    
    # 3. RANSAC to find homography matrix H
    H = find_homography(keypoints1, keypoints2, matches)
    
    # 4. Warp image to create panoramic image
    result = warp(img1, img2, H)
    
    return result

img1 = cv2.imread('data/S1.jpg')
img2 = cv2.imread('data/S2.jpg')

result = stitch_images(img1, img2)
cv2.imwrite('test.png', result)

cv2.destroyAllWindows()

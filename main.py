import cv2
from step1_2 import detect_and_describe, match_features
from step3 import homomat
from step4 import warp

def stitch_images(img1, img2):
    # 1. Interest points detection & feature description by SIFT
    keypoints1, descriptors1 = detect_and_describe(img1, "Image 1")
    keypoints2, descriptors2 = detect_and_describe(img2, "Image 2")
    
    # 2. Feature matching by SIFT features
    matches = match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2)
    
    # 3. RANSAC to find homography matrix H
    H = homomat(keypoints1, keypoints2, matches)
    
    # 4. Warp image to create panoramic image
    result = warp(img1, img2, H)
    
    return result

# img1 = cv2.imread('data/S1.jpg')
# img2 = cv2.imread('data/S2.jpg')

# result = stitch_images(img1, img2)
# cv2.imwrite('test.png', result)

# cv2.destroyAllWindows()


img1 = cv2.imread('data/S1.jpg')
img2 = cv2.imread('data/S2.jpg')
result = stitch_images(img1, img2)
cv2.imwrite('test_all.png', result)

cv2.imshow('Image', result)
cv2.waitKey(0)

cv2.destroyAllWindows()
# img1 = cv2.imread('data/TV1.jpg')
# img2 = cv2.imread('data/TV2.jpg')
# result = stitch_images(img1, img2)
# cv2.imwrite('test.png', result)
# cv2.destroyAllWindows()
# img1 = cv2.imread('data/hill1.JPG')
# img2 = cv2.imread('data/hill2.JPG')
# result = stitch_images(img1, img2)
# cv2.imwrite('test.png', result)
# cv2.destroyAllWindows()
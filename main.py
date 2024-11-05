import cv2
import os
from step1_2 import detect_and_describe, match_features
from step3 import homomat
from step4 import warp

def stitch_images(img1, img2, img1_name, img2_name):
    # 1. Interest points detection & feature description by SIFT
    keypoints1, descriptors1 = detect_and_describe(img1, img1_name)
    keypoints2, descriptors2 = detect_and_describe(img2, img2_name)
    
    # 2. Feature matching by SIFT features
    matches = match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2, output_name=f'{img1_name} & {img2_name}')
    
    # 3. RANSAC to find homography matrix H
    H = homomat(keypoints1, keypoints2, matches)
    
    # 4. Warp image to create panoramic image
    result = warp(img1, img2, H)
    
    return result

os.makedirs('output', exist_ok=True)
input_dir = 'data'
prefixes = []
for filename in os.listdir(input_dir):
    prefix = ''.join(filter(str.isalpha, os.path.splitext(filename)[0]))
    # check if prefix is not in prefixes
    if prefix not in prefixes:
        prefixes.append(prefix)

for prefix in prefixes:
    input1_path = os.path.join(input_dir, f'{prefix}1.jpg')
    img1 = cv2.imread(input1_path)
    img1_name = f'{prefix}1'
    input2_path = os.path.join(input_dir, f'{prefix}2.jpg')
    img2 = cv2.imread(input2_path)
    img2_name = f'{prefix}2'
    result = stitch_images(img1, img2, img1_name, img2_name)
    cv2.imwrite(f'./output/{prefix}1 & {prefix}2 - panorama.png', result)
    cv2.imshow(f'{prefix}1 & {prefix}2 - Panorama', result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
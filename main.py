import cv2
import numpy as np
from utils import detect_and_describe, match_features, find_homography, warp_and_stitch

# 主函數：整合各步驟
def stitch_images(img1, img2):
    # 1. 檢測和描述興趣點
    keypoints1, descriptors1 = detect_and_describe(img1, "Image 1")
    keypoints2, descriptors2 = detect_and_describe(img2, "Image 2")
    
    # 2. 匹配特徵
    matches = match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2)
    
    # 3. 找到單應矩陣 H
    H = find_homography(keypoints1, keypoints2, matches)
    
    # 4. 扭曲並拼接圖像
    result = warp_and_stitch(img1, img2, H)
    
    return result

# 加載圖像
img1 = cv2.imread('data\TV1.jpg')
img2 = cv2.imread('data\TV2.jpg')

# 拼接圖像
result = stitch_images(img1, img2)

# 結束並釋放所有窗口
cv2.destroyAllWindows()

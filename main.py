import cv2
import numpy as np

# Step 1: 使用 SIFT 進行興趣點檢測和特徵描述
def detect_and_describe(img, img_name="Image"):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 顯示檢測到的特徵點
    cv2.imshow(f"{img_name} - Keypoints", img_with_keypoints)
    cv2.waitKey(0)
    
    return keypoints, descriptors

# Step 2: 使用 SIFT 特徵進行特徵匹配
def match_features(descriptors1, descriptors2, keypoints1, keypoints2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 顯示匹配的特徵點
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Feature Matching", img_matches)
    cv2.waitKey(0)
    
    return matches

# Step 3: 使用 RANSAC 找到單應矩陣 H
def find_homography(keypoints1, keypoints2, matches, max_matches=50):
    good_matches = matches[:max_matches]  # 選取前 max_matches 個匹配點
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 顯示成功找到的單應矩陣 H
    print("Homography Matrix H:\n", H)
    
    return H

# Step 4: 計算拼接後的全景圖尺寸並進行無縫拼接
def warp_and_stitch(img1, img2, H):
    # 獲取輸入圖像的尺寸
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    
    # 計算輸出全景圖的四個頂點坐標
    corners_img1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)
    
    # 合併轉換後的圖像角點，以獲取最小邊界框
    all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 平移量
    translation_dist = [-x_min, -y_min]
    
    # 調整變換矩陣 H 以考慮平移
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    result_img = cv2.warpPerspective(img1, H_translation @ H, (x_max - x_min, y_max - y_min))
    
    # 將 img2 放置到結果圖中
    result_img[translation_dist[1]:height2 + translation_dist[1], translation_dist[0]:width2 + translation_dist[0]] = img2
    
    # 無縫拼接
    result_img = cv2.seamlessClone(img2, result_img, np.ones_like(img2) * 255, (translation_dist[0] + width2 // 2, translation_dist[1] + height2 // 2), cv2.MIXED_CLONE)
    
    # 顯示結果圖像
    cv2.imshow("Panorama", result_img)
    cv2.waitKey(0)
    
    return result_img

# 主函數：整合各步驟
def stitch_images(img1, img2):
    # 1. 檢測和描述興趣點
    keypoints1, descriptors1 = detect_and_describe(img1, "Image 1")
    keypoints2, descriptors2 = detect_and_describe(img2, "Image 2")
    
    # 2. 匹配特徵
    matches = match_features(descriptors1, descriptors2, keypoints1, keypoints2)
    
    # 3. 找到單應矩陣 H
    H = find_homography(keypoints1, keypoints2, matches)
    
    # 4. 扭曲並拼接圖像
    result = warp_and_stitch(img1, img2, H)
    
    return result

# 加載圖像
img1 = cv2.imread('data\hill1.JPG')
img2 = cv2.imread('data\hill2.JPG')

# 拼接圖像
result = stitch_images(img1, img2)

# 結束並釋放所有窗口
cv2.destroyAllWindows()

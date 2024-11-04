import cv2
import numpy as np

def detect_and_describe_mser(img, img_name="Image"):
    # MSER 檢測器
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    
    # 檢測到的 MSER 區域轉換為特徵點格式
    keypoints = []
    for region in regions:
        # 計算區域的中心點
        center = np.mean(region, axis=0)
        # 可以根據需要調整 size
        keypoints.append(cv2.KeyPoint(x=center[0], y=center[1], size=3))
    
    # SIFT 描述符
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(img, keypoints)
    
    # 顯示 MSER 檢測的特徵點
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(f"{img_name} - MSER Keypoints", img_with_keypoints)
    cv2.waitKey(0)
    
    return keypoints, descriptors

# Step 1: Interest points detection & feature description by SIFT
def detect_and_describe(img, img_name="Image"):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow(f"{img_name} - Keypoints", img_with_keypoints)
    cv2.waitKey(0)
    
    return keypoints, descriptors

# Step 2: Feature matching by SIFT features
def match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2, ratio=0.75):
    matches = []
    
    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        sorted_indices = np.argsort(distances)
        
        if distances[sorted_indices[0]] < ratio * distances[sorted_indices[1]]:
            matches.append((i, sorted_indices[0]))
    
    img_matches = np.hstack((img1, img2))
    h1, w1 = img1.shape[:2]
    
    for idx1, idx2 in matches[:50]:
        pt1 = tuple(np.round(keypoints1[idx1].pt).astype(int))
        pt2 = tuple(np.round(keypoints2[idx2].pt).astype(int) + np.array([w1, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img_matches, pt1, pt2, color, 1)
        cv2.circle(img_matches, pt1, 4, color, 1)
        cv2.circle(img_matches, pt2, 4, color, 1)
    
    cv2.imshow("Feature Matching (Custom)", img_matches)
    cv2.waitKey(0)
    
    return matches
import cv2
import numpy as np
from utils import warp_and_stitch
from step1_2 import detect_and_describe, match_features

def compute_homography(src_pts, dst_pts):
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i]
        x_prime, y_prime = dst_pts[i]
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y, -y_prime])
    A = np.array(A)

    # Use SVD to find H
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    
    H = H / H[2,2] # makes H[2,2] one
    return H

def find_homography(keypoints1, keypoints2, matches, max_matches=50):
    good_matches = matches[:max_matches]  # 選取前 max_matches 個匹配點
    src_pts = np.float32([keypoints1[m[0]].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m[1]].pt for m in good_matches]).reshape(-1, 2)

    threshold = 5.0
    K = 1000
    max_inliers = 0
    best_inliers = None
    best_H = None
    
    # 開始迭代
    for _ in range(K):
        n_points = src_pts.shape[0]

        idx = np.random.choice(src_pts.shape[0], 4, replace=False)
        H = compute_homography(src_pts[idx], dst_pts[idx])

        # Compute inliers
        src_pts_h = np.concatenate((src_pts, np.ones((n_points, 1))), axis=1)
        projected_pts = (H @ src_pts_h.T).T
        projected_pts = projected_pts[:, :2] / projected_pts[:, 2, np.newaxis]
        
        # Compute distances to dst_pts
        distances = np.linalg.norm(projected_pts - dst_pts, axis=1)
        inliers = distances < threshold
        num_inliers = np.sum(inliers)
        
        # Update the best model if the number of inliers is greater
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers = inliers
    
    # Recompute homography using all inliers for better accuracy
    if best_inliers is not None:
        best_H = compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
    
    # 顯示成功找到的單應矩陣 H
    print("Homography Matrix H:\n", best_H)
    return best_H

def stitch_images(img1, img2):
    # 1. Interest points detection & feature description by SIFT
    keypoints1, descriptors1 = detect_and_describe(img1, "Image 1")
    keypoints2, descriptors2 = detect_and_describe(img2, "Image 2")
    
    # 2. Feature matching by SIFT features
    matches = match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2)
    
    # 3. RANSAC to find homography matrix H
    H = find_homography(keypoints1, keypoints2, matches)
    
    # 4. Warp image to create panoramic image
    result = warp_and_stitch(img1, img2, H)
    
    return result

if __name__ == '__main__':
    img1 = cv2.imread('data/TV1.jpg')
    img2 = cv2.imread('data/TV2.jpg')

    result = stitch_images(img1, img2)

    cv2.destroyAllWindows()
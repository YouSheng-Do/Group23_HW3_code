import numpy as np

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

def homomat(keypoints1, keypoints2, matches, S=4, threshold=5.0, K=1000):
    src_pts = np.float32([keypoints1[m[0]].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m[1]].pt for m in matches]).reshape(-1, 2)

    max_inliers = 0
    best_inliers = None
    best_H = None
    
    # Start iteration process
    for _ in range(K):
        n_points = src_pts.shape[0]

        idxs = np.random.choice(src_pts.shape[0], S, replace=False)
        H = compute_homography(src_pts[idxs], dst_pts[idxs])

        # Compute inliers
        src_pts_h = np.concatenate((src_pts, np.ones((n_points, 1))), axis=1)
        projected_pts = (H @ src_pts_h.T).T
        projected_pts = np.array([[pts[0]/pts[2], pts[1]/pts[2]] for pts in projected_pts])

        
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
    
    # Print the best Homography matrix H
    print("Homography Matrix H:\n", best_H)
    return best_H
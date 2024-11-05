import numpy as np

def warp(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    stitch_img = np.zeros((h1, w1 + w2, 3), dtype='uint8')
    max_w = 0

    for i in range(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            cur_point = np.array([j, i, 1]).reshape(3, 1)
            transform_point = H @ cur_point
            transform_point /= transform_point[2]

            transform_x = int(transform_point[0, 0])
            transform_y = int(transform_point[1, 0])

            # with img2 parts
            if 0 <= transform_x < w2 and 0 <= transform_y < h2:
                max_w = max(max_w, j)
                stitch_img[i, j] = img2[transform_y, transform_x]

    stitch_img = stitch_img[:, :max_w]

    non_zero = np.argmax(stitch_img != 0, axis=1)
    non_zero = non_zero[non_zero > 0]
    overlap_l = np.min(non_zero)    # the left column of the overlap parts

    # linear blending
    for i in range(h1):
        for j in range(w1):
            alpha = (j - overlap_l) / (w1 - overlap_l)
            alpha = np.clip(alpha, 0, 1)
            stitch_img[i, j] = alpha * stitch_img[i, j] + (1 - alpha) * img1[i, j]

    return stitch_img
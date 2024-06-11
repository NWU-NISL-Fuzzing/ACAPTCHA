import cv2
import numpy as np


def AffineTrans(img):
    rows = img.shape[0]
    cols = img.shape[1]
    pts1 = np.float32([[20, 32], [85, 32], [20, 0]])  # 源图像中的三角形顶点坐标
    pts2 = np.float32([[20, 32], [85, 32], [40, 0]])  # 目标图像中的三角形顶点坐标
    M = cv2.getAffineTransform(pts1, pts2)  # 计算出仿射变换矩阵
    dst = cv2.warpAffine(img, M, (cols, rows))  # 应用仿射变换

    return dst


if __name__ == '__main__':
    ImageName = "F:/bishe/datasets/version/v7/train6/yyR7Y--_1637568013.png"
    img = cv2.imread(ImageName)
    affine_img = AffineTrans(img)
    cv2.imwrite("F:/bishe/datasets/version/v7/train6/yyR7Y--.png", affine_img)
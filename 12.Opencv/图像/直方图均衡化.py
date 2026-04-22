import cv2
import numpy as np

def calcAndDrawHist(image_gray):
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(hist)
    hisImg = np.zeros((256,256,3),dtype = np.uint8)
    for h in range(256):
        intensity = int(256 * hist[h][0] / maxVal)
        cv2.line(hisImg,(h,256),(h,256 - intensity),(255,0,0))
    return hisImg

image_np = cv2.imread('./lena.png')

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
hist_image = calcAndDrawHist(image_gray)

# 标准直方图均衡化
# image_equalizeHist = cv2.equalizeHist(image_gray)
# image_equalizeHist_image = calcAndDrawHist(image_equalizeHist)

# cv2.imshow('hist_image',hist_image)
# cv2.imshow('image_equalizeHist',image_equalizeHist)
# cv2.waitKey(0)

# 对比度受限的自适应直方图均衡化
clahe = cv2.createCLAHE(2,(8,8))
image_clahe = clahe.apply(image_gray)
image_clahe1 = calcAndDrawHist(image_clahe)

cv2.imshow('image_np',image_np)
cv2.imshow('image_clahe',image_clahe)
cv2.waitKey(0)

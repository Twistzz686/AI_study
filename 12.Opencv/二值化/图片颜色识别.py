import cv2
import numpy as np

# 读取图像
image_np = cv2.imread('./round.png')
print(image_np.shape)

# HSV空间转换
hsv_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

# 创建HSV范围（黄色区域）
lowerb = np.array([26, 43, 46])
upperb = np.array([34, 255, 255])

# 创建掩膜
mask_image_np = cv2.inRange(hsv_image_np, lowerb, upperb)

# 与原图进行按位与运算
color_image_np = cv2.bitwise_and(image_np, image_np, mask=mask_image_np)

# 显示结果
cv2.imshow('image_np', image_np)
cv2.imshow('mask_image_np', mask_image_np)
cv2.imshow('color_image_np', color_image_np)
cv2.waitKey()
import cv2

# 读取彩色图像
image_np = cv2.imread('./lena.png')
print(image_np.shape)

# 灰度化
image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# 二值化
ret, image_thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

# 构建腐蚀核（十字形，3x3）
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3), anchor=(0, 0))
print(kernel)

# 腐蚀操作
image_erode = cv2.erode(image_thresh, kernel)

# 显示结果
cv2.imshow('image_thresh', image_thresh)
cv2.imshow('image_erode', image_erode)
cv2.waitKey(0)

import cv2

# 读取彩色图像
image_np = cv2.imread('./lena.png')
print(image_np.shape)

# 灰度化
image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# 二值化
ret, image_thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

# 构建腐蚀核（十字形，3x3）
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3), anchor=(0, 0))
print(kernel)

# 腐蚀操作
image_erode = cv2.erode(image_thresh, kernel)

# 显示结果
cv2.imshow('image_thresh', image_thresh)
cv2.imshow('image_erode', image_erode)
cv2.waitKey(0)
import cv2


image_np = cv2.imread('./lena.png')


# 均值滤波
# image_blur = cv2.blur(image_np,(3,3))
#
# cv2.imshow('image_np',image_np)
# cv2.imshow('image_blur',image_blur)
# cv2.waitKey(0)

# 方框滤波
# image_boxFilter = cv2.boxFilter(image_np,-1,(3,3))
#
# cv2.imshow('image_np',image_np)
# cv2.imshow('image_boxFilter',image_boxFilter)
# cv2.waitKey(0)

# 高斯滤波

# image_GaussianBlur = cv2.GaussianBlur(image_np, (3, 3), 1)
#
# cv2.imshow('image_GaussianBlur', image_GaussianBlur)
# cv2.waitKey(0)

# 中值滤波

#
# image_medianBlur = cv2.medianBlur(image_np, 3)
#
# cv2.imshow('image_medianBlur', image_medianBlur)
# cv2.waitKey(0)

# 双边滤波

# image_bil = cv2.bilateralFilter(image_np,5,75,75)
#
# cv2.imshow('image_np',image_np)
# cv2.imshow('image_bil',image_bil)
# cv2.waitKey(0)
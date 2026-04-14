import cv2
import numpy as np

image_np = cv2.imread('./flower.png')
image_shape = image_np.shape

image_gray = np.zeros((image_shape[0],image_shape[1]),dtype=np.uint8)

weight_red = 0.299
weight_green = 0.587
weight_blue = 0.114

for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        image_gray[i][j] = round(image_np[i,j][0] * weight_blue +image_np[i][j][1] * weight_green + image_np[i][j][2] * weight_red)

ret, image_np_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("image_gray",image_gray)
cv2.imshow("image_np_thresh",image_np_thresh)
cv2.waitKey(0)
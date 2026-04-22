import cv2
import numpy as np

image_np = cv2.imread('./lena.png')

x_min,x_max = 150,270
y_min,y_max = 150,290

image_roi = image_np[y_min:y_max,x_min:x_max]

image_gray1 = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
image_gray2 = cv2.cvtColor(image_roi,cv2.COLOR_BGR2GRAY)

h,w = image_gray2.shape[:2]

res = cv2.matchTemplate(image_gray1,image_gray2,cv2.TM_CCOEFF_NORMED)

threshold = 0.8
location = np.where(res > threshold)


for left_top in zip(*location[::-1]):
    right_bottom = (left_top[0] + w,left_top[1] + h)
    cv2.rectangle(image_np,left_top,right_bottom,(0,0,255))

cv2.imshow('image_np',image_np)
cv2.waitKey(0)
import cv2
import numpy as np

image_np = cv2.imread('./31.png')

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

ret,image_binary = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours,hierarchy = cv2.findContours(image_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image_np,contours,-1,(0,0,255))

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    top_left = (x,y)
    bottom_right = (x+w,y+h)
    cv2.rectangle(image_np,top_left,bottom_right,(0,255,0),2)

cv2.imshow('image',image_np)
cv2.waitKey(0)

# 最小外接矩形

# for cnt in contours:
#     rect = cv2.minAreaRect(cnt)
#     box = np.int0(cv2.boxPoints(rect))
#     cv2.drawContours(image_np,[box],-1,(0,0,255),2)
#
# cv2.imshow('image',image_np)
# cv2.waitKey(0)


# 最小外接圆

# for cnt in contours:
#     (x,y),radius = cv2.minEnclosingCircle(cnt)
#     (x,y,radius) = np.int0((x,y,radius))
#     cv2.circle(image_np,(x,y),radius,(255,0,0),2)
#
# cv2.imshow('image',image_np)
# cv2.waitKey(0)
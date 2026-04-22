import cv2
import numpy as np

image_np = cv2.imread('./huofu.png')
image_shape = image_np.shape

image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

image_canny = cv2.Canny(image_gray, 30, 70)

lines = cv2.HoughLines(image_canny,0.8,np.pi / 180,90)

image_HoughLines = np.zeros(image_shape,dtype = np.uint8)

for line in lines:
    rho,theta = line[0]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x1,x2 = 0,image_shape[1]
    y1 = int((rho - x1 * cos_theta) / sin_theta)
    y2 = int((rho - x2 * cos_theta) / sin_theta)
    cv2.line(image_HoughLines,(x1,y1),(x2,y2),(0,255,0))

cv2.imshow('image',image_np)
cv2.imshow('image_HoughLines',image_HoughLines)
cv2.waitKey(0)

#概率霍夫线检测
# lines = cv2.HoughLinesP(image_canny,0.8,np.pi/180,90,minLineLength = 50,maxLineGap = 10)
#
# image_HoughLinesP = np.zeros(image_shape,dtype = np.uint8)
#
# for line in lines:
#     x1,y1,x2,y2 = line
#     cv2.line(image_HoughLinesP,(x1,y1),(x2,y2)<(0,0,255))
#
#     cv2.imshow('image_np',image_np)
#     cv2.imshow('image_HoughLinesP',image_HoughLinesP)
#     cv2.waitKey(0)

# 霍夫圆检测

# circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT,1,20,param1 =70,param2 =50)
# circles = np.int0(np.around(circles))
# image_circle = np.zeros(image_shape,dtype=np.uint8)
#
# for circle in circles:
#     x,y,radius = circle[0]
#     cv2.circle(image_circle,(x,y),radius,(0,255,0))
#
# cv2.imshow('image_circle',image_circle)
# cv2.waitKey(0)

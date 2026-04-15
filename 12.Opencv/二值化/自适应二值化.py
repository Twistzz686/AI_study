import cv2
import numpy as np

image_np = cv2.imread('./lena.png')

image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

image_adaptive = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,10)

cv2.imshow('image_adaptive',image_adaptive)
cv2.imshow('image_gray',image_gray)
cv2.waitKey(0)


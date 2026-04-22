import cv2
import numpy as np

image_np = cv2.imread('./lena.png')

image_brightness = np.uint8(np.clip(image_np * 1.0 + 20,0,255))

cv2.imshow('image_np',image_np)
cv2.imshow('image_brightness',image_brightness)
cv2.waitKey(0)
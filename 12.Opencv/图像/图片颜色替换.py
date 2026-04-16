import cv2
import numpy as np

image_np = cv2.imread('./color.png')

image_hsv = cv2.cvtColor(image_np,cv2.COLOR_BGR2HSV)

color_lower = np.array([0,43,46])
color_hight = np.array([10,255,255])
image_mask = cv2.inRange(image_hsv,color_lower,color_hight)

karnel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
image_mask_open = cv2.morphologyEx(image_mask,cv2.MORPH_OPEN,karnel)

image_np[image_mask_open == 255] = (255,0,0)
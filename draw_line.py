# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 22:30:55 2018

@author: Autumn
"""

import cv2
import math
import numpy as np

red = (0, 0, 255)  #BGR

img1 = cv2.imread('D:\\stereo\\stereo_calibration\\rectified_l.png')
img2 = cv2.imread('D:\\stereo\\stereo_calibration\\rectified_r.png')

h, w = img1.shape[:2]

tot_row = math.floor(h/10)

for r in range(tot_row):
    img1 = cv2.line(img1, (0, 20*(r-1)), (w, 20*(r-1)), red)
    img2 = cv2.line(img2, (0, 20*(r-1)), (w, 20*(r-1)), red)

img_horizontal = np.hstack((img2, img1))
img_hor_con = np.concatenate((img2, img1), axis=1)

cv2.imshow('img', img_hor_con)
cv2.waitKey(0)

cv2.destroyAllWindows()
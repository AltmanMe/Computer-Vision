# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 22:30:55 2018

@author: Autumn
"""

import cv2
import math
import numpy as np
import argparse

red = (0, 0, 255)  #BGR

def draw_line(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--left_img', required=True, help='path to the left')
    ap.add_argument('-r', '--right_img', required=True, help='path to the right')
    args = vars(ap.parse_args())
    draw_line(args['left_img'], args['right_img'])

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 19:20:51 2018

@author: Autumn
"""

import numpy as np
import cv2
import glob
import argparse
import os
from camera_calibrate import StereoCalibration

img_size = (450, 600)

def compute_disparity(filepath, savepath, M1, d1, M2, d2, R, T, img_shape, save=True):
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(M1, d1, M2, 
                                                d2, img_shape, R, T, alpha=-1)
    
    map1_x, map1_y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_shape, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_shape, cv2.CV_32FC1)
       
    img_left = glob.glob(filepath + 'left/*.jpg')
    img_right = glob.glob(filepath + 'right/*.jpg')
    
    img_left.sort()
    img_right.sort()

    for i, fname in enumerate(img_left):
        img_l = cv2.imread(img_left[i])
        img_r = cv2.imread(img_right[i])
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) 
    
        undistorted_rectified_l = cv2.remap(gray_l, map1_x, map1_y, cv2.INTER_LINEAR, img_shape)  #
        undistorted_rectified_r = cv2.remap(gray_r, map2_x, map2_y, cv2.INTER_LINEAR, img_shape)  #            
  
        max_disparity = 128
        stereo_processor = cv2.StereoSGBM_create(0, max_disparity, 21)
    
        disparity = stereo_processor.compute(undistorted_rectified_l, undistorted_rectified_r)
        cv2.filterSpeckles(disparity, 0, 4000, 128)
    
        disparity_scaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())

        cv2.imshow('img_l', undistorted_rectified_l)
        cv2.imshow('img_r', undistorted_rectified_r)

        #display disparity
        cv2.imshow('img_disparity', disparity_scaled)
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--filepath', required=True,help='filepath contains left and right file')
    ap.add_argument('-s', '--savepath', required=True, help='path to save rectified images')
    args = vars(ap.parse_args())
    
    cal_data = StereoCalibration(args['filepath'])
    camera_paras = cal_data.camera_model
    M1, d1 = camera_paras['M1'], camera_paras['d1']
    M2, d2 = camera_paras['M2'], camera_paras['d2']
    R, T = camera_paras['R'], camera_paras['T']
    
    paras = []
    paras.extend([M1, d1, M2, d2, R, T, img_size])
    
    compute_disparity(args['filepath'], args['savepaht'], *paras)
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

img_size = (600, 450)  #output images size

def callback_func(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])


def compute_disparity(filepath, M1, d1, M2, d2, R, T, img_shape, window_size):
    R1, R2, P1, P2, Q, valid_pix_roi1, _valid_pix_roi2 = cv2.stereoRectify(M1, d1, M2, 
                                                d2, img_shape, R, T, alpha=-1)
    
    map1_x, map1_y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_shape, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_shape, cv2.CV_32FC1)
       
    img_left = glob.glob(filepath + '/*.jpg')
    img_right = glob.glob(filepath + '/*.jpg')
    
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
#        window_size = 21
#        num = 5
        stereo_processor = cv2.StereoSGBM_create(0, max_disparity, window_size, 8*window_size*window_size, 32*window_size*window_size)
#        stereo_processor = cv2.StereoBM_create(16 * num, window_size)
    
        disparity = stereo_processor.compute(undistorted_rectified_l, undistorted_rectified_r)
        cv2.filterSpeckles(disparity, 0, 4000, 128)
    
        disparity_scaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())
#        disparity_scaled = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)

        cv2.imshow('img_l', undistorted_rectified_l)
        cv2.imshow('img_r', undistorted_rectified_r)

        #display disparity
        cv2.imshow('img_disparity', disparity_scaled)
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--filepath', required=True,help='path contains the left and right directory')
#    ap.add_argument('-s', '--savepath', help='path to save rectified images')
    ap.add_argument('-w', '--window_size', help='SAP WINDOW SIZE')
    args = vars(ap.parse_args())
    
    cal_data = StereoCalibration(args['filepath'])
    camera_paras = cal_data.camera_model
    M1, d1 = camera_paras['M1'], camera_paras['dist1']
    M2, d2 = camera_paras['M2'], camera_paras['dist2']
    R, T = camera_paras['R'], camera_paras['T']
    
    paras = []
    paras.extend([M1, d1, M2, d2, R, T, img_size])

    compute_disparity(args['filepath'], *paras, int(args['window_size']))

    cv2.setMouseCallback('depth', callback_func, None)
#    window_size = cv2.getTrackbarPos('SAP_Window_Size', 'depth')
#    num = cv2.getTrackbarPos('num', 'depth')


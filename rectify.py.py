# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:05:12 2018

@author: Autumn
"""

import numpy as np
import cv2
import glob
import argparse
import os
from camera_calibrate import StereoCalibration

img_size = (450, 600)

def stereo_rectify(filepath, savepath, M1, d1, M2, d2, R, T, img_shape, save=True):    
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
        
        undistorted_l = cv2.remap(img_l, map1_x, map1_y, cv2.INTER_LINEAR, img_shape)
        undistorted_r = cv2.remap(img_r, map2_x, map2_y, cv2.INTER_LINEAR, img_shape)
        
        if save:
            if not(os.path.isdir(savepath + '/left') and os.path.isdir(savepath + '/right')):
                os.mkdir(savepath + '/left')
                os.mkdir(savepath + '/right')
            cv2.imwrite(savepath + '/left/rectified_left{}.jpg'.format(i), undistorted_l)
            cv2.imwrite(savepath + '/right/rectified_right{}.jpg'.format(i), undistorted_r)
    print('Finish rectifying!!!')
            
            
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
    
    stereo_rectify(args['filepath'], args['savepaht'], *paras)
    
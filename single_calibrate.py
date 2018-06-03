# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:35:55 2018

@author: Autumn
"""

import numpy as np
import cv2
import glob
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='path to original images')
ap.add_argument('-s', '--save', help='save modified images (True or False)')
ap.add_argument('-o', '--output', help='path to save modified images')
args = vars(ap.parse_args())

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob(args['input'] + '*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if args['mode']:
    for fname in images:
        img = cv2.imread(fname)
        img_name = fname.split(os.sep)[-1].split('.')[0]
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(args['output'] + 'undistorted_{}'.format(img_name), dst)
    cv2.destroyAllWindows()
  
#calculate re-projection error
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("average re-projection error: ", tot_error/len(objpoints))   
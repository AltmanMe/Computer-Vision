# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:35:55 2018

@author: Autumn
"""

import numpy as np
import cv2
import glob
import argparse

class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        self.img_size = (450, 600)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob(cal_path + '/right/*.jpg')
        images_left = glob.glob(cal_path + '/left/*.jpg')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l2)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 6),
                                                  corners_l2, ret_l)
#                cv2.imshow(images_left[i], img_l)
#                cv2.waitKey(500*2)

            if ret_r is True:
                corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r2)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 6),
                                                  corners_r2, ret_r)
#                cv2.imshow(images_right[i], img_r)
#                cv2.waitKey(500*2)
            self.img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

        self.camera_model = self.stereo_calibrate()
        self.rms_stereo = self.camera_model['ret']
        
    # calculate mean re-projection error   
    def cal_error(self):
        tot_error_l = 0
        for i in range(len(self.objpoints)):
            imgpointsL2, _ = cv2.projectPoints(self.objpoints[i], self.r1[i], self.t1[i], self.M1, self.d1)
            error_l = cv2.norm(self.imgpoints_l[i],imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
            tot_error_l += error_l

        print("LEFT: Re-projection error: ", tot_error_l/len(self.objpoints))

        tot_error_r = 0
        for i in range(len(self.objpoints)):
            imgpointsR2, _ = cv2.projectPoints(self.objpoints[i], self.r2[i], self.t2[i], self.M2, self.d2)
            error_r = cv2.norm(self.imgpoints_r[i],imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
            tot_error_r += error_r

        print("RIGHT: Re-projection error: ", tot_error_r/len(self.objpoints))

        print(print("STEREO: RMS left to  right re-projection error: ", self.rms_stereo))

    def stereo_calibrate(self):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, self.R, self.T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', self.R)
        print('T', self.T)
        print('E', E)
        print('F', F)

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', self.R), ('T', self.T),
                            ('E', E), ('F', F), ('ret', ret)])

        cv2.destroyAllWindows()
        return camera_model
    
    def stereo_rectify(self):
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(self.M1, self.d1, self.M2, 
                                                    self.d2, self.img_shape, self.R, self.T, alpha=-1)
        
        map1_x, map1_y = cv2.initUndistortRectifyMap(self.M1, self.d1, R1, P1, self.img_shape, cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(self.M2, self.d2, R2, P2, self.img_shape, cv2.CV_32FC1)
        
        img1 = cv2.imread()
        img2 = cv2.imread()
        
        undistorted_rectified_l = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR, self.img_size)  #
        undistorted_rectified_r = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR, self.img_size)  #
        
        cv2.imwrite('rectified_l.png', undistorted_rectified_l)
        cv2.imwrite('rectified_r.png', undistorted_rectified_r)
        print('Finish writing')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filepath', help='String Filepath')
    args = vars(ap.parse_args())
    cal_data = StereoCalibration(args['filepath'])
    cal_data.camera_model

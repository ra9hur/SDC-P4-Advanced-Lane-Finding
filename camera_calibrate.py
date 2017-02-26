#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:13:01 2017

@author: raghu
"""

import numpy as np
import cv2
import glob
#import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg


def calibrate():
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    # If you are reading in an image using mpimg.imread() (matplotlib.image as mpimg) this will read in an RGB image 
    # you should convert to grayscale using cv2.COLOR_RGB2GRAY
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw the corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


    # to calibrate, calculate distortion coefficients, and test undistortion on an image
    img = cv2.imread('test_images/checker_test.jpg')
    # gray.shape[::-1]
    # img.shape[0:2]
    img_size = (img.shape[1], img.shape[0])

    # Camera calibration given object points and image points
    # mtx - camera matrix
    # dist - Distortion coeff
    # rvecs - rotation vectors
    # tvecs - translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "output_images/calibrated_params.p", "wb" ) )

    return ret

    

def calibration_check(img, nx, ny, mtx, dist):

    img_size = (img.shape[1], img.shape[0])

    # 1) Undistort using mtx and dist
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(undst, cv2.COLOR_BGR2GRAY)

    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # 4) If corners found: 
    if ret==True:
        # a) draw corners
        cv2.drawChessboardCorners(undst, (nx,ny), corners, ret)
        
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        img_8x6 = np.reshape(corners,(ny,nx,1,2))
        src = np.float32([img_8x6[0][0], img_8x6[0][7], img_8x6[5][7], img_8x6[5][0]])

        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([[100,100], [1100,100], [1100,900], [100,900]])

        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src,dst)
        
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undst, M, img_size)

    return warped, M




# 1. ------------------- Calibrate Image --------------------------------------

# Calibrate camera and save camera matrix / distortion coefficients
stat = calibrate()
print("Is camera calibrated: ", stat)


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "output_images/calibrated_params.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
checker_img = mpimg.imread('test_images/checker_test.jpg')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y


undst_checker, perspective_M = calibration_check(checker_img, nx, ny, mtx, dist)

#cv2.imwrite('tested_output/test_undist.jpg',undst_checker)
mpimg.imsave('output_images/undst_checker.jpg',undst_checker)

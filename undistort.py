#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:44:01 2017

@author: raghu
"""

#import numpy as np
import cv2
#import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
#import time as t

# https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec26/enhancing-the-contrast-in-an-image
def equalize_hist(image):

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def equalize_hist_gray(image):

    # equalize the histogram of the input image
    histeq = cv2.equalizeHist(image)

    return histeq


def rescale(data):
    data = data.astype('float32')
    return data / 255.0


def perform(image, nx, ny, mtx, dist):
    undst = cv2.undistort(image, mtx, dist, None, mtx)
    
    #undst_hist = equalize_hist(undst)

    # Apply Gaussian Smoothing -----------------------------
    #kernel_size = 3
    #blur_undst = cv2.GaussianBlur(undst_hist, (kernel_size, kernel_size), 0)

    #mpimg.imsave('extract/undst{}.jpg'.format(t.ctime()),undst)
    #image = io.imread('test_images/test{}.jpg'.format(str(i)))
    return undst

# 2. ------------------- Distort Correction -----------------------------------


if __name__ == "__main__":
    dist_pickle = pickle.load( open( "output_images/calibrated_params.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    nx = 8 # the number of inside corners in x
    ny = 6 # the number of inside corners in y

    #img = mpimg.imread('test_images/test6.jpg')
    #img = mpimg.imread('test_images/test1.jpg')
    img = mpimg.imread('test_images/test5.jpg')

    undst = perform(img, nx, ny, mtx, dist)
    #show_image(img, undst, 'Undistorted Image')
    
    mpimg.imsave('output_images/undst5.jpg',undst)


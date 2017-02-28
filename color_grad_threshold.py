#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:00:30 2017

@author: raghu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
#import pickle
import matplotlib.image as mpimg


def equalize_hist_gray(image):

    # equalize the histogram of the input image
    histeq = cv2.equalizeHist(image)

    return histeq


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = equalize_hist_gray(img[:,:,0])
    #gray = img[:,:,1]
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel_op = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    if orient=='y':
        sobel_op = cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel_op)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def gray_thresh(img, thresh=(0, 255)):

    gray = img[:,:,0]
    gray = equalize_hist_gray(gray)
    
    bin_gray = np.zeros_like(gray)
    bin_gray[(gray>thresh[0]) & (gray<=thresh[1])] = 1    

    return bin_gray


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = equalize_hist_gray(img[:,:,0])
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale_factor = np.max(abs_sobel)/255 
    scaled_sobel = (abs_sobel/scale_factor).astype(np.uint8) 

    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

    
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = equalize_hist_gray(img[:,:,0])
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 4) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(abs_sobel)
    binary_output[(abs_sobel >= thresh[0]) & (abs_sobel <= thresh[1])] = 1

    # 5) Return this mask as your binary_output image
    return binary_output

    
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    S = hls[:,:,2]
    
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S>thresh[0]) & (S<=thresh[1])] = 1
    '''
    hls = cv2.cvtColor(undst, cv2.COLOR_RGB2HLS).astype(np.float32)
    hls_yellow = cv2.inRange(hls, (10, 0, 40), (40, 200, 255))
    '''
    # 3) Return a binary image of threshold result
    return binary_output
    
    
def perform(undst):
    # Apply Gaussian Smoothing -----------------------------
    kernel_size = 3
    undst = cv2.GaussianBlur(undst, (kernel_size, kernel_size), 0)


    # --------- Thresholds for Challenge Video
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(undst, orient='x', sobel_kernel=ksize, thresh=(35, 100))
    #grady = abs_sobel_thresh(undst, orient='y', sobel_kernel=ksize, thresh=(100, 150))
    grad_gray = gray_thresh(undst, thresh=(252, 255))
    mag_binary = mag_thresh(undst, sobel_kernel=ksize, mag_thresh=(100, 150))
    #dir_binary = (dir_threshold(undst, sobel_kernel=15, thresh=(0.7, 1.3))).astype(np.float32)

    hls_binary = hls_select(undst, thresh=(100, 255))

    cg = (np.zeros_like(gradx)).astype(np.float32)
    cg[((gradx == 1) | (grad_gray == 1)) | ((mag_binary == 1))] = 1

    grad_color = np.dstack(( np.zeros_like(cg), cg, hls_binary))

    return grad_color

# 3. ------------------- Color / Gradient Threshold ---------------------------

if __name__ == "__main__":

    #undst = cv2.imread('output_images/undst6.jpg')
    #undst = mpimg.imread('output_images/undst1.jpg')
    #undst = mpimg.imread('output_images/undst5.jpg')
    undst = mpimg.imread('output_images/undst11.jpg')
    #undst = mpimg.imread('output_images/undst12.jpg')
    #undst = mpimg.imread('output_images/undst13.jpg')
    #undst = mpimg.imread('output_images/undst14.jpg')
    #undst = mpimg.imread('output_images/undst15.jpg')
    #undst = mpimg.imread('output_images/undst16.jpg')
    #undst = mpimg.imread('output_images/undst17.jpg')
    #undst = mpimg.imread('output_images/undst18.jpg')
    #undst = mpimg.imread('output_images/undst19.jpg')

    # Apply Gaussian Smoothing -----------------------------
    #kernel_size = 3
    #undst = cv2.GaussianBlur(undst, (kernel_size, kernel_size), 0)

    grad_color = perform(undst)
    plt.imshow(grad_color,'gray')

    mpimg.imsave('output_images/grad_color.jpg',grad_color)
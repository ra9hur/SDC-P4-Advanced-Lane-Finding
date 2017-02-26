#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:11:45 2017

@author: raghu
"""

import numpy as np
import cv2
import matplotlib.image as mpimg


# Retain only the area that contains lanes and discard all other pixels
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon formed from `vertices`. 
    The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Given thresholded image, return a binary warped image
def warp_perspective(img,src,dst):

    img_size = (img.shape[1], img.shape[0])

    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src,dst)
        
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M


# Helper function to draw lines on an image
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` - Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def draw_lanes(img, src_points):
    """
    Returns an image with lines drawn.
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    pts = src_points.reshape((-1,1,2))
    cv2.polylines(line_img,[pts],True,color=[255, 0, 0], thickness=3)

    combo =  weighted_img(line_img, img, α=0.8, β=1., λ=0.)
    return combo
    

# This is the main method that gets called by the pipeline
def perform(grad_color):
    
    # . Select Region Of Selection ------------------- 
    ysize = grad_color.shape[0]
    xsize = grad_color.shape[1]

    left_bottom = [90, ysize]
    right_bottom = [xsize-90, ysize]
    apex = [xsize*0.5, ysize*0.57]      # 0.57
    vertices = np.int32([left_bottom, right_bottom, apex])
    vertices = vertices.reshape((1,-1,2))
    region_select = region_of_interest(grad_color, vertices)

    
    # . Source Points using Hough Transform ------------------- 
    
    src_int = np.int32([[585,460], [203,720], [1127,720], [695,460]])
    src_flt = np.float32([[585,460], [203,720], [1127,720], [695,460]])
        
    img_lane = draw_lanes(grad_color, src_int)

    
    # . Perspective Transform ------------------- 

    dst_int = np.int32([[320,0], [320,720], [960,720], [960,0]])
    dst_flt = np.float32([[320,0], [320,720], [960,720], [960,0]])
    
    binary_warp, M = warp_perspective(region_select, src_flt,dst_flt)
    binary_unwarp, Minv = warp_perspective(region_select, dst_flt,src_flt)

    binary_warped_lane = draw_lanes(binary_warp, dst_int)
    
    result = {}
    result["region_select"] = region_select
    result["binary_warp"] = binary_warp
    result["M"] = M
    result["binary_unwarp"] = binary_unwarp
    result["Minv"] = Minv
    result["img_lane"] = img_lane
    result["binary_warped_lane"] = binary_warped_lane
    
    return result
    
    
# . ------------------- Perspective Transform ---------------------------------

if __name__ == "__main__":

    #grad_color = mpimg.imread('output_images/grad_color6.jpg')
    #grad_color = mpimg.imread('output_images/grad_color1.jpg')
    grad_color = mpimg.imread('output_images/grad_color5.jpg')

    pt_params = perform(grad_color)

    binary_warp = pt_params["binary_warp"]

    #M = pt_params["M"]
    #binary_unwarp = pt_params["binary_unwarp"]
    Minv = pt_params["Minv"]
    img_lane = pt_params["img_lane"]
    binary_warped_lane = pt_params["binary_warped_lane"]

    #show_image(img_lane, binary_warped_lane, 'Binary Warped Image')    

    #mpimg.imsave('output_images/binary_warp5.jpg',binary_warp)

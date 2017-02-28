#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:53:11 2017

@author: raghu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# To remove outliers and replace by median
def replace_outliers(signal, threshold=5):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)    #replace outliers with median
    return signal


# Compute bottom X values for left / right lanes
def compute_lowxvals(left_fit,right_fit):
    y_low = 720.
    left_lowxvals = left_fit[0]*(y_low**2) + left_fit[1]*y_low + left_fit[2]
    right_lowxvals = right_fit[0]*(y_low**2) + right_fit[1]*y_low + right_fit[2]
    return left_lowxvals,right_lowxvals


# Compute left offset from the center of the image
def center_offset(left_fit_new,right_fit_new):
    
    left_lowxvals,right_lowxvals = compute_lowxvals(left_fit_new,right_fit_new)
    
    lane_center = (right_lowxvals - left_lowxvals)/2.
    image_center = 1280./2
    offset = image_center - lane_center

    # Convert pixel offset to meters
    xm_per_pix = 3.7/700.       # meters per pixel in x dimension
    offset *= xm_per_pix

    return offset


# Populate X values to left/right line instances
def compute_xvals(left_fit_new,right_fit_new,left_line,right_line):

    left_lowxvals,right_lowxvals = compute_lowxvals(left_fit_new,right_fit_new)

    m = len(left_line.lastnx)
    n = len(right_line.lastnx)

    if (m>5):
        left_line.lastnx.pop(0)
        left_line.lastnx.append(left_lowxvals)
        left_line.prevx = left_line.bestx
        left_line.bestx = np.median(left_line.lastnx)
    else:
        left_line.lastnx.append(left_lowxvals)
        left_line.prevx = left_line.bestx
        left_line.bestx = left_lowxvals

    if (n>5):
        right_line.lastnx.pop(0)
        right_line.lastnx.append(right_lowxvals)
        right_line.prevx = right_line.bestx
        right_line.bestx = np.median(right_line.lastnx)
    else:
        right_line.lastnx.append(right_lowxvals)
        right_line.prevx = right_line.bestx
        right_line.bestx = right_lowxvals

    return left_line,right_line


# Validate and update best fit values to left/right line instances
def compute_bestfit(left_line,right_line,left_minpix,right_minpix):

    # ----- Check for valid radius of curvature
    lr = left_line.radius_of_curvature
    rr = right_line.radius_of_curvature
    
    good_rad = True
    if ((lr/rr > 10.) | (rr/lr > 10.)):
        good_rad = False

    good_lxval = False
    if (300 < left_line.bestx < 425.):
        good_lxval = True

    m = len(left_line.lastn_fit)
    n = len(right_line.lastn_fit)
    if ((200 < left_line.radius_of_curvature < 10000.) & (left_minpix) & (good_rad) & (good_lxval)):
        left_line.detected.pop(0)
        left_line.detected.append(True)
        if (m>10):
            left_avg = np.median(left_line.lastn_fit, axis=0)
            left_line.best_fit = 0.3*left_line.current_fit + 0.7*left_avg
            left_line.lastn_fit.pop(0)
        else:
            left_line.best_fit = left_line.current_fit
        left_line.lastn_fit.append(left_line.current_fit)
    else:
        left_line.detected.pop(0)
        left_line.detected.append(False)


    if ((200 < right_line.radius_of_curvature < 10000.) & (right_minpix) & (good_rad)):
        right_line.detected.pop(0)
        right_line.detected.append(True)
        if (n>10):
            right_avg = np.median(right_line.lastn_fit, axis=0)
            right_line.best_fit = 0.3*right_line.current_fit + 0.7*right_avg
            right_line.lastn_fit.pop(0)
        else:
            right_line.best_fit = right_line.current_fit
        right_line.lastn_fit.append(right_line.current_fit)
    else:
        right_line.detected.pop(0)
        right_line.detected.append(False)

    return left_line,right_line


# Fit line for the initial frame
def fit_line_init(binary_warped,left_line,right_line):
    #print("Calling fit_line_init")
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    # binary_warped has 3 channels, convert to gray to create histogram
    gray_warped = cv2.cvtColor(binary_warped,cv2.COLOR_RGB2GRAY)
    histogram = np.sum(gray_warped[np.int32(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.copy(binary_warped)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    leftx = replace_outliers(leftx, threshold=3)
    lefty = replace_outliers(lefty, threshold=3)
    rightx = replace_outliers(rightx, threshold=3)
    righty = replace_outliers(righty, threshold=3)

    left_line.allx = leftx
    left_line.ally = lefty
    right_line.allx = rightx
    right_line.ally = righty
    
    # Fit a second order polynomial to each
    left_line.current_fit = np.polyfit(lefty, leftx, 2)
    right_line.current_fit = np.polyfit(righty, rightx, 2)

    left_curverad, right_curverad = curvature(leftx,lefty,rightx,righty)
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad

    left_line.line_base_pos = center_offset(left_line.current_fit,right_line.current_fit)
    right_line.line_base_pos = left_line.line_base_pos

    # set x values
    left_lowxvals,right_lowxvals = compute_lowxvals(left_line.current_fit,right_line.current_fit)
    left_line.prevx = left_lowxvals
    left_line.bestx = left_lowxvals
    right_line.prevx = right_lowxvals
    right_line.bestx = right_lowxvals

    left_line,right_line = compute_bestfit(left_line,right_line,True,True)

    return (out_img,left_line,right_line)


# Fit line for subsequent frames
def fit_increment(binary_warped,left_line,right_line):
    #print("Calling fit_increment")
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # replace outliers with median values
    leftx = replace_outliers(leftx, threshold=3)
    lefty = replace_outliers(lefty, threshold=3)
    rightx = replace_outliers(rightx, threshold=3)
    righty = replace_outliers(righty, threshold=3)

    left_curverad, right_curverad = curvature(leftx,lefty,rightx,righty)

    left_fit_new = np.polyfit(lefty, leftx, 2)
    right_fit_new = np.polyfit(righty, rightx, 2)
    left_line.line_base_pos = center_offset(left_fit_new,right_fit_new)
    right_line.line_base_pos = left_line.line_base_pos

    pixel_thresh = 10000
    left_minpix = True
    right_minpix = True

    if (len(leftx)<pixel_thresh):
        left_minpix = False

    if (len(rightx)<pixel_thresh):
        right_minpix = False

    left_line,right_line = compute_xvals(left_fit_new,right_fit_new,left_line,right_line)

    left_line.allx = leftx
    left_line.ally = lefty
    left_line.current_fit = left_fit_new
    left_line.diffs = left_fit_new - left_fit
    left_line.radius_of_curvature = left_curverad

    right_line.allx = rightx
    right_line.ally = righty
    right_line.current_fit = right_fit_new
    right_line.diffs = right_fit_new - right_fit
    right_line.radius_of_curvature = right_curverad

    left_line,right_line = compute_bestfit(left_line,right_line,left_minpix,right_minpix)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.copy(binary_warped)
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return (result,left_line,right_line)


# Compute radii of curvature for left/right lanes
def curvature(leftx,lefty,rightx,righty):

    # Define conversions in x and y from pixels space to meters
    y_eval = 720            # 720p video/image
    ym_per_pix = 30/720     # meters per pixel in y dimension
    xm_per_pix = 3.7/700    # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


# Draw lanes on the undistorted image and visualize
def overlay(binary_warped, undst, left_line, right_line, Minv):

    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_lowxvals,right_lowxvals = compute_lowxvals(left_fit,right_fit)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(binary_warped, np.int_([pts]), (0,255, 0))
    cv2.polylines(binary_warped,np.int_([pts]),True,color=[255, 0, 0], thickness=3)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(binary_warped, Minv, (undst.shape[1], undst.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undst, 1, newwarp, 0.3, 0)

	# Add 'lane curvature' and 'vehicle offset from center' values to the image
    label_str = 'Radius of left line curvature: %.1f m' % left_line.radius_of_curvature
    result = cv2.putText(result, label_str, (30,40), 0, 1, (255,0,0), 2, cv2.LINE_AA)

    label_str = 'Radius of right line curvature: %.1f m' % right_line.radius_of_curvature
    result = cv2.putText(result, label_str, (30,80), 0, 1, (255,0,0), 2, cv2.LINE_AA)
    
    label_str = 'Vehicle offset from lane center: %.1f m' % float(left_line.line_base_pos)
    result = cv2.putText(result, label_str, (30,120), 0, 1, (255,0,0), 2, cv2.LINE_AA)

    label_str = 'X position of left line: %.1f pixels' % float(left_lowxvals)
    result = cv2.putText(result, label_str, (30,160), 0, 1, (255,0,0), 2, cv2.LINE_AA)

    label_str = 'X position of right line: %.1f pixels' % float(right_lowxvals)
    result = cv2.putText(result, label_str, (30,200), 0, 1, (255,0,0), 2, cv2.LINE_AA)

    return result


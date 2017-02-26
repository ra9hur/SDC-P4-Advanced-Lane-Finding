#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:28:19 2017

@author: raghu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import undistort
import color_grad_threshold
import perspective_transform
import lanes_overlay

import logging
from importlib import reload
reload(logging)

LOG_FILENAME = '/home/raghu/src/CarND-Advanced-Lane-Lines/debug/check.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

left_line = None
right_line = None


def show_image(org, updated, upd_title):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.tight_layout()
    ax1.imshow(org)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(updated, 'gray')
    ax2.set_title(upd_title, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def equalize_hist(img):

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    cv2.waitKey(0)
    return img_output


def normalized(img):
    return np.uint8(255*img/np.max(np.absolute(img)))


def to_RGB(img):
   if img.ndim == 2:
       img_normalized = normalized(img)
       return np.dstack((img_normalized, img_normalized, img_normalized))
   elif img.ndim == 3:
       return img
   else:
       return None


def compose_diagScreen(curverad=0, offset=0, mainDiagScreen=None,
                     diag1=None, diag2=None, diag3=None, diag4=None, diag5=None, diag6=None, diag7=None, diag8=None, diag9=None):
      # middle panel text example
      # using cv2 for drawing text in diagnostic pipeline.
      #font = cv2.FONT_HERSHEY_COMPLEX
      #middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
      #cv2.putText(middlepanel, 'Estimated lane curvature: {}'.format(curverad), (30, 60), font, 1, (255,0,0), 2)
      #cv2.putText(middlepanel, 'Estimated Meters right of center: {}'.format(offset), (30, 90), font, 1, (255,0,0), 2)

      # assemble the screen example
#      diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
      diagScreen = np.zeros((720, 1920, 3), dtype=np.uint8)
      if mainDiagScreen is not None:
            diagScreen[0:720, 0:1280] = mainDiagScreen
      if diag1 is not None:
            diagScreen[0:240, 1280:1600] = cv2.resize(to_RGB(diag1), (320,240), interpolation=cv2.INTER_AREA) 
      if diag2 is not None:
            diagScreen[0:240, 1600:1920] = cv2.resize(to_RGB(diag2), (320,240), interpolation=cv2.INTER_AREA)
      if diag3 is not None:
            diagScreen[240:480, 1280:1600] = cv2.resize(to_RGB(diag3), (320,240), interpolation=cv2.INTER_AREA)
      if diag4 is not None:
            diagScreen[240:480, 1600:1920] = cv2.resize(to_RGB(diag4), (320,240), interpolation=cv2.INTER_AREA)*4
      if diag7 is not None:
            diagScreen[600:1080, 1280:1920] = cv2.resize(to_RGB(diag7), (640,480), interpolation=cv2.INTER_AREA)*4
#      diagScreen[720:840, 0:1280] = middlepanel
#      if diag5 is not None:
#            diagScreen[840:1080, 0:320] = cv2.resize(to_RGB(diag5), (320,240), interpolation=cv2.INTER_AREA)
#      if diag6 is not None:
#            diagScreen[840:1080, 320:640] = cv2.resize(to_RGB(diag6), (320,240), interpolation=cv2.INTER_AREA)
#      if diag9 is not None:
#            diagScreen[840:1080, 640:960] = cv2.resize(to_RGB(diag9), (320,240), interpolation=cv2.INTER_AREA)
#      if diag8 is not None:
#            diagScreen[840:1080, 960:1280] = cv2.resize(to_RGB(diag8), (320,240), interpolation=cv2.INTER_AREA)

      return diagScreen


class Line():
    NX = 8              # the number of inside corners in x
    NY = 6              # the number of inside corners in y
    MTX = None
    DIST = None
    FIRST_FRAME = True
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = [True,True,True,True]  
        # x values of the last n fits of the line
        #self.recent_xfitted = [] 
        self.lastnx = []
        #average x values of the fitted line over the last n iterations
        self.prevx = None
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.lastn_fit = []
        # best fit computed considering average and current fit
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        #self.current_fit = [np.array([False])]  
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


def process_image(img):
    global left_line, right_line
    
    undst = undistort.perform(img, Line.NX, Line.NY, Line.MTX, Line.DIST)

    #undst = equalize_hist(undst)


    # . ------------------- Color / Gradient Threshold ------------------------
    grad_color = color_grad_threshold.perform(undst)
    mpimg.imsave('output_images/grad_color.jpg',grad_color)
    grad_color = mpimg.imread('output_images/grad_color.jpg')


    # . ------------------- Perspective Transform -----------------------------
    pt_params = perspective_transform.perform(grad_color)

    #region_select = pt_params["region_select"]
    binary_warp = pt_params["binary_warp"]
    #M = pt_params["M"]
    #binary_unwarp = pt_params["binary_unwarp"]
    Minv = pt_params["Minv"]
    #img_lane = pt_params["img_lane"]
    #binary_warped_lane = pt_params["binary_warped_lane"]
    #show_image(img_lane,binary_warped_lane,"Top view of the road")
    
    # . ------------------- Detect Lane Lines:Initial test6.jpg ---------------
    if Line.FIRST_FRAME:
        out_img,left_line,right_line = lanes_overlay.fit_line_init(binary_warp,left_line,right_line)
        Line.FIRST_FRAME = False
    else:
        out_img,left_line,right_line = lanes_overlay.fit_increment(binary_warp,left_line,right_line)


    # . ------------------- Compute Lane Curvature ----------------------------

    logging.debug(left_line.detected)
    logging.debug(left_line.prevx)
    logging.debug(left_line.bestx)
    logging.debug(left_line.radius_of_curvature)
    logging.debug(left_line.line_base_pos)
    logging.debug(right_line.detected)
    logging.debug(right_line.prevx)
    logging.debug(right_line.bestx)
    logging.debug(right_line.radius_of_curvature)
    #logging.debug(left_line.lastn_fit)
    #logging.debug(right_line.lastn_fit)
    #logging.debug(left_line.best_fit)
    #logging.debug(right_line.best_fit)
    #logging.debug(left_line.current_fit)
    #logging.debug(right_line.current_fit)
    #logging.debug(left_line.diffs)
    #logging.debug(right_line.diffs)
    #logging.debug(len(left_line.allx))
    #logging.debug(len(right_line.allx))
    logging.debug('*'*20)


    if (int(left_line.radius_of_curvature) == int(right_line.radius_of_curvature)):
        Line.FIRST_FRAME = True
    
    if (np.count_nonzero(left_line.detected == 0) | np.count_nonzero(right_line.detected == 0)):
        Line.FIRST_FRAME = True

    # . ------------------- Drawing the lines back down onto the road ---------
    # Create an image to draw the lines on
    result = lanes_overlay.overlay(binary_warp, undst, left_line, right_line, Minv)

    #curverad = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2.0
    
    #diagScreen = compose_diagScreen(curverad, left_line.line_base_pos, result,
    #                 undst, region_select, grad_color, binary_warped_lane)

    return result



# . ------------------- Helper Functions End ----------------------------------
# . ---------------------------------------------------------------------------



if __name__ == "__main__":


    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "output_images/calibrated_params.p", "rb" ))
    Line.MTX = dist_pickle["mtx"]
    Line.DIST = dist_pickle["dist"]

    #image = cv2.imread('test_images/test6.jpg')
    #image = mpimg.imread('test_images/test1.jpg')
    #image = mpimg.imread('test_images/test5.jpg')
    #image = cv2.imread('output_images/undst6.jpg')
    #image = mpimg.imread('test_images/test1.jpg')
    #image = mpimg.imread('test_images/test5.jpg')
    #image = mpimg.imread('test_images/test11.jpg')
    #image = mpimg.imread('test_images/test12.jpg')
    #image = mpimg.imread('test_images/test13.jpg')
    #image = mpimg.imread('test_images/test14.jpg')
    #image = mpimg.imread('test_images/test15.jpg')
    #image = mpimg.imread('test_images/test16.jpg')
    #image = mpimg.imread('test_images/test17.jpg')
    #image = mpimg.imread('test_images/test18.jpg')
    #image = mpimg.imread('test_images/test19.jpg')

    # Instantiate lines
    left_line = Line()
    right_line = Line()


    # . ------------------- Overlay lanes on the video / image ----------------
    white_output = 'output_video.mp4'
    #clip1 = VideoFileClip("project_video.mp4")
    clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


    #final_image = process_image(image)
    #plt.imshow(final_image)

    '''
    # . ------------------- Distort Correction --------------------------------
    undst = undistort.perform(image, Line.NX, Line.NY, Line.MTX, Line.DIST)
    #advlaneutil.show_image(img, undst, 'Undistorted Image')
    #mpimg.imsave('output_images/undst6.jpg',undst6)


    # . ------------------- Color / Gradient Threshold ------------------------
    grad_color = color_grad_threshold.perform(undst)
    mpimg.imsave('output_images/grad_color.jpg',grad_color)
    grad_color = mpimg.imread('output_images/grad_color.jpg')

#    grad_color = np.unit8(grad_color)
    #advlaneutil.show_image(undst, grad_color, 'Color/Gradient Image')
    #mpimg.imsave('output_images/grad_color.jpg',grad_color)
    #grad_color = mpimg.imread('output_images/grad_color.jpg')


    # . ------------------- Perspective Transform -----------------------------

    pt_params = perspective_transform.perform(grad_color)

    binary_warp = pt_params["binary_warp"]
    M = pt_params["M"]
    binary_unwarp = pt_params["binary_unwarp"]
    Minv = pt_params["Minv"]
    img_lane = pt_params["img_lane"]
    binary_warped_lane = pt_params["binary_warped_lane"]
    
    #binary_warp, M, binary_unwarp, Minv, img_lane, binary_warped_lane = perspective_transform.perform(grad_color)
    #show_image(img_lane, binary_warped_lane, 'Binary Warped Image')


    # . ------------------- Detect Lane Lines:Initial test6.jpg ---------------
    #binary_warped6 = mpimg.imread('output_images/binary_warped6.jpg')
    #binary_warp = mpimg.imread('output_images/binary_warp.jpg')
    left_line = Line()
    right_line = Line()
    
    if Line.FIRST_FRAME:
        # out_img,left_fit,right_fit,leftx,lefty,rightx,righty = lanes_overlay.fit_line_init(binary_warp)
        out_img,left_line,right_line = lanes_overlay.fit_line_init(binary_warp,left_line,right_line)
        Line.FIRST_FRAME = False
    else:
        out_img,left_line,right_line = lanes_overlay.fit_increment(binary_warp,left_line,right_line)
        # out_img,left_fit,right_fit,leftx,lefty,rightx,righty = lanes_overlay.fit_increment(binary_warp,left_fit,right_fit)

    '''
    '''
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warp.shape[0]-1, binary_warp.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    #    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    plt.imshow(out_img)
    plt.imshow(binary_warp)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    '''
    '''
    # . ------------------- Compute Lane Curvature ----------------------------
    # Extract left and right line pixel positions
    # left_curverad, right_curverad = curvature(ploty, leftx, rightx)
    #left_curverad, right_curverad = lanes_overlay.curvature(leftx, rightx, lefty, righty)
    # Both of above lines yield same result

    #print(left_curverad, 'm', right_curverad, 'm')
    
    left_line,right_line = lanes_overlay.curvature(left_line,right_line)
    # Both of above lines yield same result
    print('left_line: ',left_line.radius_of_curvature, 'm\n', 'right_line: ',right_line.radius_of_curvature, 'm')


    # . ------------------- Drawing the lines back down onto the road ---------
    # Create an image to draw the lines on
    #result = lanes_overlay.overlay(binary_warp, undst, left_fitx, right_fitx, ploty, Minv)
    result = lanes_overlay.overlay(binary_warp, undst, left_line, right_line, Minv)
    plt.imshow(result)

    '''
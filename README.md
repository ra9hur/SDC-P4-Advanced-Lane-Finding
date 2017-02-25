# SDC-P4-Advanced-Lane-Finding
Project deals with complex scenarios like curving lines, shadows and changes in the color of the pavement.

1. Problem Definition - About Project
-------------
In the very first project, we implemented a basic pipeline to find lane lines on the road.

This project implements a better lane finding algorithm that deals with complex scenarios like curving lines, shadows and changes in the color of the pavement.
In addition, lane curvatures and vehicle offsets with respect to the center are measured. These are the kinds of measurements we will ultimately need to make in order to control the car.

|#                     |Project4 - Advanced Lane Finding	      |Project 1 – Lane Finding     
|:-------:			  |:----               	            |:---
|1                |Compute camera calibration matrix & distortion coeff given, set of chessboard images.        |     
|2		|Apply a distortion correction to raw images.	|Apply gray scale to the image; followed by Gaussian smoothing
|3 	|Use color transforms, gradients, etc., to create a thresholded binary image. |Apply Canny, an optimized auto thresholding edge detector that takes an image and based on local values, determine a threshold to create a single pixel thick edge
|4 |Select region of interest to retain lane pixels in the lower half of the image |Select region of interest to retain lane pixels in the lower half of the image
|5 |Apply a perspective transform to rectify binary image ("birds-eye view") - used for detecting perspectively transformed shapes. |Hough Tranform line detection - used for detecting translated shapes.
|6 |Detect lane pixels & fit to find lane boundary. |Detect line segments in the image, then extrapolate
|7 |Determine the curvature of the lane and vehicle position with respect to center. |
|8 |Draw the detected lane boundaries back onto the original image. |Draw the detected lane boundaries back onto the original image.
|9 |Output visual display of the lane boundaries with numerical estimation of lane curvature and vehicle position. |Output visual display of the lane boundaries

Most of the code for this project is leveraged from the lecture notes.

This project is step1 of the overall implementation

1. First, the advanced lane-finding (Project 4)
2. Second, the vehicle detection and tracking (Project 5)

With this, it can be perceived that Udacity is trying to subtly introduce students to syllabus in term 2 (simultaneous localization and mapping – SLAM) using camera sensors.

----------

2. How to run
-------------
**Order of executing files to view image output**

1. camera_calibrate
2. undistort
3. color_grad_threshold
4. perspective_transform
5. lanes_overlay 	-> 	To be invoked from ‘advlanelines.py’

To execute the pipeline on an image and visualize final output, run ‘advlanelines.py’

**To generate video output**

Run python ‘advlanelines.py’. This will take the raw video file 'project_video.mp4', and creates an  output video 'output_video.mp4'. To run the lane detector on arbitrary video files, update corresponding lines of ‘advlanelines.py’.

----------

3. Camera Calibration
-------------

To compute the transformation between 3D object points in the world and 2D image points.

**Why is this step required ?**

Cameras don't create perfect images. Some of the objects in the images, especially ones near the edges, can get stretched or skewed in various ways and we need to correct such distortions.
In order to get perspective transformation right, we first have to correct for the effect of image distortion.

**Implementation**

The camera was calibrated using the chessboard images in 'camera_cal/*.jpg'. 
The following steps were performed for each calibration image:

- Convert to grayscale
- Find chessboard corners with OpenCV's findChessboardCorners() function, assuming a 8x6 board
- After the above steps were executed for all calibration images, OpenCV's calibrateCamera() function is used to calculate the distortion matrices. 
- Using the distortion matrices, images are undistorted using OpenCV's undistort() function.

**Verification**

*1. Have the camera matrix and distortion coefficients been computed correctly and checked on one of the calibration images as a test?*

![undst_checker](https://cloud.githubusercontent.com/assets/17127066/23333841/0511cf54-fbb9-11e6-927c-7536673132e6.png)

**Code / output_images**

1. The code to perform camera calibration is in 'camera_calibrate.py'
2. The final calibration matrices are saved in the pickle file 'calibrated_params.p'

----------

 4. Distortion Correction
-------------
To ensure that the geometrical shape of objects is represented consistently, no matter where they appear in an image.

**Why is this step required ?**

We are trying to accurately place the self-driving car in this world. Eventually, we'll want to look at the curve of a lane and steer the correct direction. But if the lane is distorted, we'll get the wrong measurement for curvature in the first place and our steering angle will be wrong.

**Verification**

*2. Has the distortion correction been correctly applied to each image?*

Below is the example undistorted image.

![undst5](https://cloud.githubusercontent.com/assets/17127066/23333840/0500d316-fbb9-11e6-8166-ec4ff55bb13d.jpg)

**Code / output_images**

1. The code to perform distortion correction is in 'undistort.py'. 
2. Camera calibration matrices in 'calibrate_camera.p' is used to undistort the input image. 
3. For all images in 'test_images/*.jpg', the undistorted version of that image is saved in 'output_images/undst*.png'.

----------

 5. Thresholded binary image
-------------
**Why is this step required ?**

Canny is an optimised auto-thresholding edge detector, great at finding all possible lines in an image. But for lane detection, this gave us a lot of edges on scenery, cars and other objects that we ended up discarding. Knowing the fact that lanes that we are looking for, are close to vertical, it would be effective to detect steep edges that are more likely to be lanes in the first place by using multiple gradient threshold combinations. 

Also, exploring color spaces other than RGB enhances the ability to detect lanes in shadows and changes in the color of the pavement.

**Implementation**

The goal is to identify pixels that are likely to be part of the lane lines.

- Input -  undst*.png
- Apply the following filters with thresholding, to create separate "binary images" corresponding to each individual filter
	- Absolute horizontal Sobel operator on the image using G component of RGB
	- Absolute vertical Sobel operator on the image using G component of RGB
	-  Image transformed to R component of RGB and histogram is equalized using the cv2.equalizeHist() function
	- Sobel operator in both horizontal and vertical directions and calculate its magnitude  using R component of RGB
	- Sobel operator to calculate the direction of the gradient using R component of RGB
	- Convert the image from RGB space to HLS space, and threshold the S channel
- Combine the above binary images to create the final binary gradient and color thresholded image
- Output -  grad_color*.png

**Verification**

*3. Has a binary image been created using color transforms, gradients or other methods?*
Here is the example image, transformed into a binary image by combining the above thresholded binary filters:

![grad_compare](https://cloud.githubusercontent.com/assets/17127066/23333836/04e46d52-fbb9-11e6-9341-f3cf22d8cd3b.png)

**Code / output_images**

1. The code to generate the thresholded binary image is in 'color_grad_threshold.py'.
2. For all images in 'test_images/{}.jpg', the thresholded binary version of that image is saved in 'output_images/grad_color{}.png'.


----------

6. Perspective transform
-------------
To transform an image such that we are effectively viewing objects from a different angle or direction. 
In an image, perspective is the phenomenon where an object appears smaller, the farther away it is from a viewpoint like a camera, and parallel lines appear to converge to a point. 
Considering perspective in the image of the road, the lane looks smaller and smaller, the farther away it gets from the camera, and the background scenery also appears smaller than the trees closer to the camera in the foreground.
A perspective transform uses this information to transform an image.
It essentially transforms the apparent z coordinate (depth) of the object points, which in turn changes that object's 2D image representation.
A perspective transform warps the image and effectively drags points towards or pushes them away from the camera to change the apparent perspective.

**Why is this step required ?**

To steer a car, we'll need to measure how much, the  lane is curving. Parallel lanes when converted from 3D world image to 2D camera image appear to be curving and also converging as they go further. To retrieve actual representation in 3D, we need to map out the lanes in camera images, after transforming them to a different perspective. One way is to get the top-view i.e. having a view of the road from above.
Doing a bird's-eye view transform is especially helpful for road images because it will also allow us to match a car's location directly with a map, since maps display roads and scenery from a top down view.

**Implementation**

- Input - grad_color*.png
- Select the region in the original image that is most likely to have the lane line pixels.
- The goal is to transform the image such that we get a "bird's eye view" of the lane, which enables us to fit a curved line to the lane lines (e.g. polynomial fit). 
- To accomplish the perspective transform, OpenCV's getPerspectiveTransform() and warpPerspective() functions are used. Source and destination points are hard-coded for the perspective transform. The source and destination points were visually determined by manual inspection. Those same four source points should work to transform any image (under the assumption that the road is flat and the camera perspective hasn't changed).
- Output - binary_warp*.png'

**Verification**

*4. Has a perspective transform been applied to rectify the image?*

Here is the example image, after applying perspective transform:

![binary_warp_compare](https://cloud.githubusercontent.com/assets/17127066/23333835/04dd2f24-fbb9-11e6-9ce5-4b29069af5e5.png)

**Code / output_images**

1. The code to perform perspective transform is in 'perspective_transform.py'
2. For all images in 'test_images/*.jpg', the warped version of that image is saved in 'output_images/binary_warp*.png'.

----------

7. Polynomial fit
-------------
**Why is this step required ?**

To detect lanes in complex scenarios like curving lines. This step takes binary_warped image as input, determines pixels that are lane line pixels, determine the line shape and the position.

Given the warped binary image from the previous step, a 2nd order polynomial is fit to both left and right lane lines.

**Initial frame**

- Input: binary_warp, initialized left_line / right_line instances
- Calculate a histogram of the bottom half of the image
- Partition the image into 9 horizontal slices
- Split the histogram in half vertically
- Starting from the bottom slice, enclose a 200 pixel wide window around the left peak and right peak of the histogram 
- Go up the horizontal window slices to find pixels that are likely to be part of the left and right lanes, recentering the sliding windows opportunistically
- Given 2 groups of pixels (left and right lane line candidate pixels), fit a 2nd order polynomial to each group, which represents the estimated left and right lane lines
- The code to perform the above is in the fit_line_init() function of 'lanes_overlay.py'.
- Output: updated left_line / right_line instances

**Successive frame**

There exists a temporal correlation between two successive frames. Polynomial parameters found from the previous frame is utilized to fit the line in subsequent frames

- binary_warp, updated left_line / right_line instances
- Plot lines from parameters from the previous frame
- Consider a window margin 100 around these lines. Pixels within this window margin are considered to to fit polynomial for the current frame
- Perform a 2nd order polynomial fit to those pixels found from our quick search. 
- The code to perform an incremental search is in the fit_increment() function of 'lanes_overlay.py'.

**Sanity Check**

A lot of parameters are tracked through line instances and this becomes handy to check correlation between frames

- In case we don't find enough pixels for the lanes (minpix = 10000), lane lines from the previous frame is retained and the current frame is ignored.
	- line_detected set to True
- Pixel outliers within the window margin are identified and then replaced by the median 
- Check, if bottom X values change drastically from previous frames
	- Xvals for previous 10 frames are stored in line instances.
	- Average of previous Xvals is compared with the current value.
	- Any change < 50 pixels is considered good
- Tracking changes to vehicle distance from the center
	- If the offset changes > 0.3m, measurements are considered bad
- If radius of curvature of lanes < 250m and > 10000 m, measurements are considered bad and lane parameters for the current frame is dropped
- If the ratios of radius of curvature of lanes is > 10, measurement for the current frame is skipped
- If either left / right lane is not detected for 3 consecutive frames, parameters are reset and polynomials are fit through the initial histogram method.

**Verification**

*5. Have lane line pixels been identified in the rectified image and fit with a polynomial?*
Below is an illustration of the output of the polynomial fit, for our original example image. 

![polyfit_compare](https://cloud.githubusercontent.com/assets/17127066/23333839/04fa2a84-fbb9-11e6-89fb-2010b5c64618.png)

**Code / output_images**

1. The code to perform perspective transform is in 'lanes_overlay.py'
2. For all images in 'test_images/*.jpg', the polynomial-fit-annotated version of that image is saved in 'output_images/polyfit_*.png'.


----------

8. Overlay original image with lane area
-------------
Given all the above, we can annotate the original image with the lane area, and information about the lane curvature and vehicle offset.

- Create a blank image, and draw our polyfit lines (estimated left and right lane lines)
- Fill the area between the lines (with green color)
- Use the inverse warp matrix calculated from the perspective transform, to "unwarp" the above such that it is aligned with the original image's perspective
- Overlay the above annotation on the original image
- Add text to the original image to display lane curvature and vehicle offset

![overlay_compare](https://cloud.githubusercontent.com/assets/17127066/23333838/04f4f460-fbb9-11e6-9aca-dd3e2f9beb99.png)

**Code / output_images**

1. The code to perform the above is in the function overlay() in 'lanes_overlay.py'.
2. Image with visual display of lane boundary is saved in 'output_images/polyfit_*.png'.

----------

9. Radius of curvature
-------------
**Why is this step required ?**

As the lanes are curving - left or right, steering wheel should be turned with the required angle. This angle can be determined by knowing speed/dynamics of the car and the radius of curvature of lanes.
To determine the curvature, following steps are followed.

1. Detect the lane lines using some masking and thresholding techniques.
2. Perform a perspective transform to get a birds eye view of the lane.
3. Fit a polynomial to the lane lines which couldn't have been done easily before.
4. Extract the curvature of the lines from this polynomial with just a little math.

**Implementation**

Input: Polynomial fit for the left and right lane lines
Radius of curvature for each line can be calculated according to formula, ....
Distance units are converted from pixels to meters, assuming 30 meters per 720 pixels in the vertical direction, and 3.7 meters per 700 pixels in the horizontal direction.
Average of radius of curvature for the left and right lane lines are taken

**Verification**

*6. Having identified the lane lines, has the radius of curvature of the road been estimated?*

**Code / output_images**

The code to calculate the radius of curvature is in the function curvature() in 'lanes_overlay.py'.

----------

10. Vehicle offset from lane center
-------------
**Why is this step required ?**

This is measured to understand the current position of vehicle, avoid swaying towards lanes and then bring it towards the center.

**Implementation**

- Input: Polynomial fit for the left and right lane lines
- To calculate the vehicle's offset, lane center is assumed as the center of the image. Lane center is calculated as the mean x value of the bottom x value of the left lane line, and bottom x value of the right lane line. The offset is simply the vehicle's center x value (i.e. center x value of the image) minus the lane's center x value.
- Given the polynomial fit for the left and right lane lines, I calculated the vehicle's offset from the lane center. The vehicle's offset from the center is annotated in the final video. I made the same assumptions as before when converting from pixels to meters.

**Verification**

*7. Determine position of the vehicle with respect to center in the lane*

**Code / output_images**

The code to calculate the vehicle's lane offset is in the function center_offset() in 'lanes_overlay.py'.

----------

11. Pipeline (video)
-------------
**Verification**

*8. Does the pipeline established with the test images work to process the video?*

----------

12. Fine-tuning
-------------
1. Temporal correlation is exploited to smooth-out the polynomial fit parameters. The benefit to doing so would be to make the detector more robust to noisy input. Simple moving average across frames were used to take mean of the polynomial coefficients (3 values per lane line) for the most recent 10 video frames.

2. Diagnosis Tool

![overlay5_diagnosys](https://cloud.githubusercontent.com/assets/17127066/23333837/04ec4d42-fbb9-11e6-897c-1ecfa25fb459.png)

This project involves fine tuning of lot of parameters like color thresholding, gradient thresholding values to obtain the best lane detection. This can be trickier if the pipeline fails for few video frames. To efficiently debug this, a new frame was built that captures multiple stages of the pipeline, like the original image, color/gradient thresholding, region selected and binary_warped frames.
Thanks to John Chen for sharing this tool

	Ref: https://carnd-forums.udacity.com/questions/32706990/want-to-create-a-diagnostic-view-into-your-lane-finding-pipeline

3. With the pipeline developed for project_video when applied for challenge_video, there are too many edges detected. Gradient threshold had to be changed to just consider below operators
	- Absolute horizontal Sobel operator on the image using G component of RGB
	-  Image transformed to R component of RGB and histogram is equalized using the cv2.equalizeHist() function
	- Convert the image from RGB space to HLS space, and threshold the S channel

----------

13. Further Improvements 
------------- 

- The project involves processing and extracting features manually. The process is too time-consuming. Also, it is always possible that the pipe-line tuned for certain conditions may not work in other situations. Would like to explore if this project can be implemented through deep-learning.
- In challenge_video, there are too many vertical edges and the algorithm often wrongly detects one of the edges as lanes. This is isolated to certain extent with thresholding. However, there is still scope to get better results
- The developed pipeline fails to work with harder_challenge_video. Need to explore if there is a different way to solve this problem.

----------

##Writeup Template
Kelly Smith

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[distort]: ./output_images/distorted_checkerboard.png "Distorted"
[image1]: ./output_images/undistort_car.png "Undistorted"
[image3]: ./output_images/binary_image.png "Binary Example"
[image4]: ./output_images/straight_topdown.png "Warp Example"
[lanePixelID]: ./output_images/lane_pixel_identification.png "Fit Visual"
[medianFiltering]: ./output_images/median_filtering.png "Rejecting Spurious Measurements"
[lanePoly]: ./output_images/lane_polynomials.png "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the **Undistort Demonstration** cell of the IPython notebook located in `P4_Advanced_Lane_Finding.ipynb`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Original, distorted image][distort]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I originally implemented my own function to create a thresholded binary image called `create_binary_img()`, but I was never able to obtain satisfactory performance from it.  To benchmark against it, I implemented the method provided in the lecture material, and it performed far better than my original method.

The code is located in the section of the Jupyter Notebook called **Create binary image from color thresholding and edge detection**.

The Udacity-provided method performs a bitwise `OR` to combine multiple binary layers together.  The binary layers are formed from thresholding applied to a saturation color layer (from a Hue-Lightness-Saturation color decomposition) and a horizontal edge detection (Sobel-x) layer.  I adjusted the thresholds on the Saturation layer to obtain better performance.

Here is an example binary image created with this function. 

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform was in a function called `transform_perspective` in my Jupyter Notebook.

The function `transform_perspective` accepts the input image `img` to be transformed, a perspective transformation matrix $$ M $$, and the destination image shape.  The matrix $$ M $$ is stored as a global variable in the notebook, and it's computed by the `get_perspective_transform()` function during the `setup()` function.  

The `get_perspective_transform()` function calls the OpenCV function `cv2.getPerspectiveTransform()` which accepts source and destination vertices.

For my source vertices, I defined them by picking points from the test images provided with the project.  I found a set of points that approximately mapped straight lane lines into straight _vertical_ lines in the transformed image.

The final set of source points I used was:

```
warpSrcPoints = np.float32([ [545, 450], [725, 450], [1280, 720], [0, 720] ])
```

I defined the destination points as:

```
imgShape = [720, 1280]
inset = 100
warpDstPoints = np.float32([ [inset,0], [imgShape[1]-inset, 0], 
    [imgShape[1]-inset,imgShape[0]], [inset,imgShape[0]] ])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the section **Lane Pixel Identification** in my Jupyter Notebook, I've implemented the functionality to detect lane-line pixels. 

The algorithm handles the left and right lane detection separately.  The left lane is handled first, but both lanes are handled nearly identically.

First, the algorithm computes a histogram of the bottom half of the warped image.  This histogram will generally have two prominent peaks which correspond to the left and right lane markers.  I identify the index of the peak left-lane marker by using the `np.argmax()` function on the left half of the histogram.  I perform the same task for the right lane by processing the right half of the histogram.  These initial peaks define my **coarse** solution which is used to initialize the **fine** solution search algorithm.

The fine solution algorithm works by dividing the image into 9 horizontal stripes.  Using the prior solution (or the coarse solution on the first iteration), a search window is established about that point.  The width of the search window is defined is 1/5 of the width of the overall image, and the height is defined to lie within the current horizontal stripe.  The **fine** search process works by computing the histogram over the small search window.  If enough lane pixels exist within the search window, then the peak is identified within the search window.  As long as the location of the new peak does not differ too greatly from the previous peak, then it is accepted as the new peak.  

If there are not enough lane pixels detected in the existing search window, then the algorithm simply outputs a stale $$x$$-coordinate of the previous stripe's $$x$$-coordinate.  I had attempted a solution to expand the search area, but it would occasionally lead to failure of the primary algorithm, so I removed that logic.


This process repeats over all the horizontal stripes in the image for the left and right lane lines. 

The output of all this is a set of $$ (x,y) $$ coordinates which I will refer to as **key points** for each lane that represent the lane pixels.  The key points are shown as yellow dots in the image below.

![alt text][lanePixelID]

**Filtering Code**

These $$ (x,y) $$ coordinates are recorded by the `Line` objects which records the last $$N$$ sets of key points from the last $$N$$ frames of the video.  Upon saving the key points into the `Line` object, it will fit a quadratic polynomial to the keypoints and then interpolate the polynomial across the image.  These $$M$$ interpolated points are saved for each stored frame of video.

Next, for each frame, the median interpolated value at each sampled $$y$$-value is computed.  Effectively, this is taking the median value of the fit polynomials at each $$y$$-sample.

Finally, a 2nd-order polynomial fit is derived from the set of median keypoints.  These are the curves that are used to create the polygon overlay on the road.

As implemented, the technique is currently designed to retain the key points from the last 10 frames of video.  This has the effect of introducing some lag into the lane-detection, and this lag is most noticeable when the vehicle is pitching up and down.  One can notice the overlay temporarily lagging behind the actual lane markers for a few frames.  This lag is a natural consequence of the fact that we are smoothing the raw measurement signal with prior stale solutions.  However, the benefit of this technique is that it performs well in rejecting spurious lane detections over a few frames.  Below is a plot of a test where spurious key points were incorporated over 5 frames.  The raw key points are shown as connected red dots and the polynomial fit to the median key points is shown in black. 

![alt text][medianFiltering]

Here is a plot of the fit polynomials.

![alt text][lanePoly]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in my Jupyter Notebook in the functions `computePositionError()`, `computeLaneCurvature()`, and `computeLaneCurvaturePx()`.  

I computed the position error in the `computePositionError()` function by measuring the difference between the image center (image width divided by 2) and the apparent lane center. The apparent lane center is computed by finding the average of the nearest lane pixel on the left and right.  For instance, if the the left lane's closest key point was at `x=200` and the right lane's closest key point was at `x=1000`, then the lane center would be `x=600`.  Since the image width is 1280 pixels wide, the image center is at `x=640`.  This implies that the image center is 40 pixels to the right of the lane center.  I used the conversion ratio to convert this pixel difference into meters.

Next, I compute the lane curvature in real-space (meters) using the `computeLaneCurvature()` function, which is based on the Udacity lecture material.

Next, I compute the lane curvature in pixel space (pixels) using the `computeLaneCurvaturePx()` function.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the `processImg()` function in my code in the section titled **Executive Pipeline Function**.

Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./submission.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project was challenging because it required so much iterative tuning to get each step working well.  Once the pipeline was performing well for the test images, running the pipeline on the video revealed the challenging spots (shadows on roadway, changes in pavement color).

The binary image thresholding algorithm seems to lack robustness.   Rather than performing a bitwise `OR` across the various layers, it would be better to somehow combine the layers in a probabilistic fashion (convolving each layer together) and then normalizing.  This would help to mitigate the effects of a single layer encountering problems.  For instance, the edge detection channel may report a lot of false positives due to shadows in the roadway, but the Saturation channel may successfully reject those shadows.  Due to the disagreement, those binary pixels should be treated with more suspicion and have a lower probability of surviving into the final output binary image.

The lane-pixel-finding algorithm is also brittle.  This algorithm begs for a Kalman filter or a particle filter to help estimate and track the position of the lane lines from image to image.  I've implemented a simple smoothing technique with rudimentary rejection of spurious measurements, but a more fully-featured solution like residual editing from Kalman filtering would be simpler to maintain with fewer edges cases.

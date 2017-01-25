import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os.path

"""
Define constants for pipeline
"""

# Camera Calibration filename 
calibrationFilename = "CameraCalibration.p"

# Warping Source Points
warpSrcPoints = np.float32([ [580, 450], [705, 450], [1115, 700], [200, 700] ])

# Warp Destination Points
imgShape = [720, 1280]
inset = 100
warpDstPoints = np.float32([ [inset,inset], [imgShape[1]-inset, inset], 
	[imgShape[1]-inset,imgShape[0]-inset], [inset,imgShape[0]-inset] ])
	

def pipeline(Recalibrate=False, ProcessImages=False):
	# Do all the things in sequence.

	# Obtain camera coeffiicents
	mtx, dist = get_camera_calibration_data(Recalibrate)

	# Get perspective transform matrix M
	# Obtain the perspective transformation (should be moved outside the loop)
	M = get_perspective_transform([720, 1280])

	# Define thresholds
	thresholds = {'SobelX':[0, 255], 'hls':[100, 255]}

	# img = mpimg.imread('test_images/test1.jpg')
	ProcessImages = True
	if ProcessImages:
		test_images = glob.glob('test_images/*.jpg')
		for testImg in test_images:

			img = mpimg.imread(testImg)

			# Resize all images for consistency
			img = cv2.resize(img, (1280,720))
			
			# Process image using pipeline 
			processImg(img, mtx, dist, M, thresholds)
			



def processImg(img, mtx, dist, M, thresholds):
	# Undistort the image
	undistortedImg = undistort_img(img, mtx, dist)

	# Apply thresholding (color transforms, gradients) to obtain binary image
	
	thresholded = create_binary_img(undistortedImg, thresholds)
	
	# Warp the image with a perspective transform
	birdsEyeView = transform_perspective(thresholded, M, (thresholded.shape[1], thresholded.shape[0]))

	windowPeak = findLanePixels(birdsEyeView)

	# Identify the lane pixels in the bird's eye view image
	## laneImg = findLanePixels()

	# Fit a 2nd-order polynomial to the lane lines
	## leftLane, rightLane = fitLane()

	# Compute the lane curvature
	## curvature = computeLaneCurvature()

	# Compute the position error (off centerline)
	## posError = computeLaneCurvature()

	# Plot results
	f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,9))
	f.tight_layout()
	ax1.imshow(img)
	for i in range(len(warpSrcPoints)):
		ax1.plot(warpSrcPoints[i][0], warpSrcPoints[i][1],'.')

	ax1.set_title('Original Image',fontsize=50)
	ax2.imshow(thresholded, cmap='gray')
	ax2.set_title('Thresholded', fontsize=50)
	ax3.imshow(birdsEyeView, cmap='gray')
	for i in range(len(warpSrcPoints)):
		ax3.plot(warpDstPoints[i][0], warpDstPoints[i][1],'.')
	ax3.set_title('Birds Eye View', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	for i in range(len(windowPeak)):
		for j in range(2):
			ax3.plot(windowPeak[i][j][0], windowPeak[i][j][1],'s')
	plt.show()


"""
Manager function that either calibrates or loads existing calibration data if it exists
"""
def get_camera_calibration_data(Recalibrate):
	# Calibrate the camera, if commanded by the user or if no pickled data exists.
	if Recalibrate or not os.path.isfile(calibrationFilename):
		print("Calibrating camera...",end='')
		chessBoardSize = (9,6)
		mtx, dist = calibrateCamera(chessBoardSize)
		print("Complete!")
		print("mtx = " + str(mtx))
		print("dist = " + str(dist))

		calibration_test(mpimg.imread('camera_cal/calibration10.jpg'), mtx, dist)
		# calibration_test(mpimg.imread('calibration_wide/test_image.jpg'), mtx, dist)
	else:
		print("Loading existing caliberation data...", end='')
		x = loadCalibrationData()
		mtx = x['mtx']
		dist = x['dist']
		print("Complete!")
	return mtx, dist


"""
Code to calibrate the camera (computing distortion coefficients)
"""
def calibrateCamera(chessBoardSize):
	

	images = glob.glob('camera_cal/calibration*.jpg')
	# images = glob.glob('calibration_wide/GOPR*.jpg')

	objpoints = []
	imgpoints = []

	objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2) # x, y coordinates

	ctr = 1
	n = len(images)
	for filename in images:
		print("(%d/%d)" % (ctr, n))
		ctr += 1
		# read in each image
		img = mpimg.imread(filename)

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# Find chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, chessBoardSize, None)

		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)
		else:
			print("WARNING: Calibration on %s failed." % (filename))

	# Calibrate the camera
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	# Save the camera calibration data
	saveCalibrationData(mtx, dist, rvecs, tvecs)
	
	# Return the Camera matrix (mtx) and Distortion Coefficients (dist)
	return (mtx, dist)

"""
Serialize the camera calibration data via pickle
"""
def saveCalibrationData(mtx, dist, rvecs, tvecs):
	x = {"mtx":mtx, "dist":dist}
	# calibrationFilename = "CameraCalibration.p"
	with open (calibrationFilename,"wb") as output_file:
		pickle.dump(x, output_file)
		print("Saved the Camera Calibration data to %s" % calibrationFilename)

"""
Loads serialized camera calibration data
"""
def loadCalibrationData():
	with open (calibrationFilename,"rb") as input_file:
		x = pickle.load(input_file)
	return x

def calibration_test(img, mtx, dist):
	undistorted = undistort_img(img, mtx, dist)

	f, (ax1, ax2) = plt.subplots(1,2,figsize=(24,9))
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image',fontsize=50)
	ax2.imshow(undistorted)
	ax2.set_title('Undistorted Image', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()


"""
Undistort the image using camera calibration.
"""
def undistort_img(img, mtx, dist):
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)
	return undistorted

def create_binary_img(img, thresholds):
	# Use color transforms, gradients to get thresholded binary image
	
	# Layers
	# 1. Grayscale Sobel-x edge detection
	# 2. Directed edge detetions
	# 3. RSS of Sobel-x and Sobel-y
	# 4. Image Saturation layer

	# Convert to grayscale
	
	# raw_sobelx = sobelx(img)
	# Apply thresholds
	# binary_sobelx = raw_sobelx
	# binary_sobelx = np.zeros_like(raw_sobelx)
	# binary_sobelx[(binary_sobelx >= thresholds['SobelX'][0]) & (binary_sobelx <= thresholds['SobelX'][1])] = 1

	binary_hls = hlsSelect(img, thresholds['hls'])
	binary_sobel = mag_thresh(img)



	# Perform Sobel-x and take absolute value to find edges
	# sobelx8 = sobelx(gray)

	# Apply threshold
	# sxbinary = np.zeros_like(sobelx8)
	# sxbinary[(sobelx8 >= thresholds[0]) & (sobelx8 <= thresholds[1])] = 1

	# Merge layers
	merged_binary = cv2.bitwise_or(binary_hls, binary_sobel)
	# merged_binary = np.zeros_like(img)
	# merged_binary[(binary_hls==1) | (sobelx==1)] = 1

	return merged_binary

def hlsSelect(img, thresholds):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]

	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresholds[0]) & (s_channel <= thresholds[1])] = 1
	return binary_output


def sobelx(img):
	# Perform Sobel-x and take absolute value to find edges
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sobelx_raw = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	# Convert to 8-bit
	return np.uint8(255*sobelx_raw/np.max(sobely_raw))

def sobely(img):
	# Perform Sobel-x and take absolute value to find edges
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sobely_raw = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Convert to 8-bit
	return np.uint8(255*sobely_raw/np.max(sobely_raw))

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    # binary_output = np.zeros_like(gradmag)
    # binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return gradmag

def get_perspective_transform(imgShape):
	x_centerline = imgShape[1]/2
	horizon_lane_width = imgShape[1]*0.08
	x_left_horizon = x_centerline - horizon_lane_width/2.0
	x_right_horizon = x_centerline + horizon_lane_width/2.0
	y_horizon = 450
	
	# Order of points is top-left, top-right, bottom-right, bottom-left
	# Define source points (picked by hand)
	#src = np.float32([ [x_left_horizon, y_horizon], [x_right_horizon, y_horizon], 
	#				   [imgShape[1], imgShape[0]], [0, imgShape[0]] ])
	#src = np.float32([ [580, 440], [712, 440], 
	#				   [1115, 700], [200, 700] ])
	
	# Define destination points (maps to rectangle)
	#inset = 100
	#dst = np.float32([ [inset,inset], [imgShape[1]-inset, inset], [imgShape[1]-inset,imgShape[0]-inset], [inset,imgShape[0]+inset] ])
	
	M = cv2.getPerspectiveTransform(warpSrcPoints, warpDstPoints)
	return M

"""
Perform image warping
"""
def transform_perspective(img, M, dest_img_size):
	# Transform image to birds-eye-view
	warped = cv2.warpPerspective(img, M, dest_img_size) # , flags=cv2.INTER_LINEAR)
	return warped

def findLanePixels(img, numYBins=10):
	
	# Coarse: Identify the peaks in the bottom half of the image.
	histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
	
	xSplit = len(histogram)/2
	leftPeak = np.argmax(histogram[:xSplit])
	rightPeak = np.argmax(histogram[xSplit:])+xSplit

	print("----- COARSE -------")
	print("Left peak is at %d" % (leftPeak))
	print("Right peak is at %d" % (rightPeak))
	
	windowWidth = img.shape[1] / 10

	# Perform the sliding window method to identify all points
	yBins = np.linspace(img.shape[0], 0, numYBins)

	print("------ FINE --------")
	windowPeak = []
	leftLaneMarker = []
	rightLaneMarker = []
	for i in range(numYBins-1):

		# Define bounds for sliding window
		yMax = yBins[i+1]
		yMin = yBins[i]

		# Left
		xMin = max([leftPeak - windowWidth/2, 0])
		xMax = xMin + windowWidth
		print("Left Window: %d < X < %d   |   %d < Y < %d  " % (xMin, xMax, yMin, yMax))
		hist = np.sum(img[yMin:yMax, xMin:xMax], axis=0)
		plt.plot(hist)
		print(img[yMin:yMax, xMin:xMax])
		if max(hist) != 0:
			print("No pixels detected!")
			leftPeak = np.argmax(hist) + xMin

		print("Left: Max(hist) = %d" % (max(hist)))
		
		leftLaneMarker.append([leftPeak, (yMax+yMin)/2])

		# Right
		xMax = min([rightPeak + windowWidth/2, img.shape[1]])
		xMin = xMax - windowWidth
		
		print("Right Window: %d < X < %d   |   %d < Y < %d  " % (xMin, xMax, yMin, yMax))
		hist = np.sum(img[yMin:yMax, xMin:xMax], axis=0)
		print("Right: Max(hist) = %d" % (max(hist)))
		rightPeak = np.argmax(hist) + xMin
		rightLaneMarker.append([rightPeak, (yMax+yMin)/2])

	print(leftLaneMarker)
	print(rightLaneMarker)
	return [leftLaneMarker, rightLaneMarker]


	

def computeLaneCurvature():
	pass

def computePositionError():
	pass

def warpBoundaries():
	pass

pipeline()



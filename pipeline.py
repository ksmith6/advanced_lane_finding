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

def test():
	pass

def pipeline(Recalibrate=False, ProcessImages=False):
	# Do all the things in sequence.

	# Obtain camera coeffiicents
	mtx, dist = get_camera_calibration_data(Recalibrate)

	# Get perspective transform matrix M
	img = mpimg.imread('test_images/test1.jpg')
	M = get_perspective_transform(img.shape)
	birdsEyeView = transform_perspective(img, M, (img.shape[1], img.shape[0]))
	plt.imshow(birdsEyeView)
	plt.show()
	
	if ProcessImages:
		test_images = glob.glob('test_images/*.jpg')
		for testImg in test_images:
			# Undistort the image
			undistortedImg = undistort_img(img, mtx, dist)

			# Apply thresholding (color transforms, gradients) to obtain binary image
			# binaryImg = create_binary_img(undistortedImg)

			# Warp the image with a perspective transform
			birdsEyeView = transform_perspective(undistortedImg, M, undistortedImg.shape)

			# Identify the lane pixels in the bird's eye view image
			## laneImg = findLanePixels()

			# Fit a 2nd-order polynomial to the lane lines
			## leftLane, rightLane = fitLane()

			# Compute the lane curvature
			## curvature = computeLaneCurvature()

			# Compute the position error (off centerline)
			## posError = computeLaneCurvature()


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

def create_binary_img(img):
	# Use color transforms, gradients to get thresholded binary image
	
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	return img


def get_perspective_transform(imgShape):
	x_centerline = imgShape[1]/2
	horizon_lane_width = imgShape[1]*0.2
	x_left_horizon = x_centerline - horizon_lane_width/2.0
	x_right_horizon = x_centerline + horizon_lane_width/2.0
	y_horizon = 450
	
	# Order of points is top-left, top-right, bottom-right, bottom-left
	# Define source points (picked by hand)
	src = np.float32([ [x_left_horizon, y_horizon], [x_right_horizon, y_horizon], 
					   [imgShape[1], imgShape[0]], [0, imgShape[0]] ])
	
	# Define destination points (maps to rectangle)
	dst = np.float32([ [0,0], [imgShape[1], 0], [imgShape[1],imgShape[0]], [0,imgShape[0]] ])
	
	M = cv2.getPerspectiveTransform(src, dst)
	return M

"""
Perform image warping
"""
def transform_perspective(img, M, dest_img_size):
	# Transform image to birds-eye-view
	warped = cv2.warpPerspective(img, M, dest_img_size) # , flags=cv2.INTER_LINEAR)
	return warped

def findLanePixels():
	# Identify lanes
	pass

def computeLaneCurvature():
	pass

def computePositionError():
	pass

def warpBoundaries():
	pass

pipeline()



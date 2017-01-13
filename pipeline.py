import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

def test():
	pass

def pipeline():
	# Do all the things in sequence.
	chessBoardSize = (9,6)
	mtx, dist = calibrateCamera(chessBoardSize)
	print("mtx = " + str(mtx))
	print("dist = " + str(dist))

	calibration_test(mpimg.imread('camera_cal/calibration10.jpg'), mtx, dist)


	

"""
Code to calibrate the camera (computing distortion coefficients)
"""
def calibrateCamera(chessBoardSize):
	
	images = glob.glob('camera_cal/calibration*.jpg')
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
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, chessBoardSize, None)

		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)
		else:
			print("WARNING: Calibration on %s failed." % (filename))

	
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	if ret==True:
		saveCalibrationData(mtx, dist, rvecs, tvecs)
	else:
		print("WARNING: Camera Calibration failed!")

	# Return the Camera matrix (mtx) and Distortion Coefficients (dist)
	return (mtx, dist)

def saveCalibrationData(mtx, dist, rvecs, tvecs):
	x = {"mtx":mtx, "dist":dist}
	calibrationFilename = "CameraCalibration.p"
	with open (calibrationFilename,"wb") as output_file:
		pickle.dump(x, output_file)
		print("Saved the Camera Calibration data to %s" % calibrationFilename)

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

def create_binary_img():
	# Use color transforms, gradients to get thresholded binary image
	pass


def perspectiveTransform():
	# Transform image to birds-eye-view
	pass

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



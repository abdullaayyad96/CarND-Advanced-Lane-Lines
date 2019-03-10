import numpy as np
import matplotlib.image as mpimg
import cv2
import sys
import pickle
import os

def calibrate(directory, size):
    #This functions return the camera matrix and distortion coefficients by perfroming calibration on a set of chessboard images
    #obtaining files in directory
    cal_files = os.listdir(directory)

    #defining image and object points
    img_points = []
    obj_points = []
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1,2)

    #iterating over images in directory 
    for file in cal_files:
        
        #reading image
        if(file[-3:]=="jpg"):
            cal_img = mpimg.imread(directory + file)

            #converting to grayscale
            gray = cv2.cvtColor(cal_img, cv2.COLOR_RGB2GRAY)

            #obtaining corners in chessboard
            ret, corners = cv2.findChessboardCorners(gray, size)
            if (ret):
                img_points.append(corners)
                obj_points.append(objp)

    #performing calibration
    ret, cam_mtx, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return cam_mtx, dist_coef
	
	
def main(argvs):

	if len(argvs) > 1:
		calibrate_img_dir = argvs[1]
		
		if calibrate_img_dir[-1] != "/":
			calibrate_img_dir += "/"
	else:
		print('Please provide directory of calibration images')
		sys.exit()		
	
	#initialize
	chess_size = (9, 6) #size of chessboard in calibration images
	mtx = np.ndarray(shape=(3,3)) #setting camera matrix as global variables
	dist = np.ndarray(shape=(1,5))  #setting distortion coefficients as global variables


	#perform calibration
	[mtx, dist] = calibrate(calibrate_img_dir, chess_size)
	
	#save calibration parameters
	cal_mtx_dir = "cal_mtx.sav"
	cal_dist_dir = "cal_dist_dir"
	pickle.dump(mtx, open(cal_mtx_dir, 'wb'))
	pickle.dump(dist, open(cal_dist_dir, 'wb'))
	
main(sys.argv)
sys.exit()
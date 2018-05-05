import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import os
import sys
from Line import Line
from moviepy.editor import VideoFileClip

#set video or image mode
mode = 'video'
#images directory
img_dir = 'test_images/'
img_out_dir = 'output_images/'
#videos directory
video_dir = 'project_video.mp4'
video_dir_out = 'output_project_video.mp4'

#calibration settings
calibrate_b = True #boolean to determine whether to perform calibration
calibrate_img_dir = 'camera_cal/' #directory of calibration files1
chess_size = (9, 6) #size of chessboard in calibration files
mtx = np.ndarray(shape=(3,3)) #setting camera matrix as global
dist = np.ndarray(shape=(1,5))  #setting distortion coefficients as global

#undistortion settings 
undistort_b = True

#perspective transform settings 
#These points were obtained manually
source_points = np.float32([ [594, 450], [688, 450], [280, 675], [1040, 675] ])
destination_points = np.float32([ [250, 0], [1030, 0], [250, 720], [1030, 720] ])

#thresholding settings
sobel_thresholding = True
color_thresholding = True
sobel_mag_thresh = (30, np.inf)
sobel_dir_thresh = (0.8, 2.3)
color_s_thresh = (160, 255)
color_l_thresh = (160, 255)

#lane finding settins
#type of sliding window, convolution or regular
slide_mode = 'regular'

#Create instance of line class
myLine = Line()

def calibrate(directory, size):
    #obtaining files in directory
    cal_files = os.listdir(directory)

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


def undistort(img, camera_mtx, dist_coef):
    #return undistorted images
    return cv2.undistort(img, camera_mtx, dist_coef)

def perspective_transform(img, src, dst):
    #obtain transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    #performing and returning perspective transform
    return cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
 

def threshold(img, color_s_thresh=(150, 255), color_l_thresh=(100, 255), sobel_mag_thresh=(50, 255), sobel_dir_thresh=(0.7, 1.3)):
    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) #sobel in x direction
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) #sobel in y direction
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    abs_sobely = np.absolute(sobely) # Absolute y derivative to accentuate lines away from vertical

    sobel_dir = np.arctan2(abs_sobelx, abs_sobely)
    sobel_mag = np.sqrt(np.square(sobelx)+np.square(sobely))
    scaled_sobel_mag =  np.uint8(255*sobel_mag/np.max(sobel_mag))

    sobel_dir_binary = np.zeros_like(s_channel)
    sobel_dir_binary[(scaled_sobel_mag >= sobel_dir_thresh[0]) & (scaled_sobel_mag <= sobel_dir_thresh[1])] = 1

    sobel_mag_binary = np.zeros_like(s_channel)
    sobel_mag_binary[(sobel_dir >= sobel_mag_thresh[0]) & (sobel_dir <= sobel_mag_thresh[1])] = 1

    sobel_binary = np.zeros_like(s_channel)
    sobel_binary[ (sobel_dir_binary==1) & (sobel_mag_binary==1)] = 1
        
    ## Threshold x gradient
    #sxbinary = np.zeros_like(scaled_sobel)
    #sxbinary[(scaled_sobel > sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1
    
    ## Threshold color s channel
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= color_s_thresh[0]) & (s_channel <= color_s_thresh[1])] = 1

    ##threshold color l channel
    #l_binary = np.zeros_like(s_channel)
    #l_binary[(s_channel >= color_l_thresh[0]) & (s_channel <= color_l_thresh[1])] = 1

    ##combining l and s thresholds
    #ls_binary = np.zeros_like(l_binary)
    #ls_binary[ (l_binary == 1) & (s_binary == 1)] = 1

    ls_channel = l_channel + s_channel
    ls_binary = np.zeros_like(ls_channel)
    ls_binary[ (ls_channel >= (color_s_thresh[0] + color_l_thresh[0])) & (ls_channel <= (color_s_thresh[1] + color_l_thresh[1]))] = 1

    # Stack each channel
    # be beneficial to replace this channel with something else.
    color_binary = np.uint8(np.dstack(( np.zeros_like(s_channel), ls_binary, sobel_binary))) * 255
    combined_binary = np.zeros_like(ls_binary)
    combined_binary[ (ls_binary == 1) | (sobel_binary == 1)] = 1
    return combined_binary, color_binary


def find_lines(binary_warped, Line):
    #Finding the lines by utilizing a sliding window method and returing fitted polynomials

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Set the width of the windows +/- margin
    margin = 100
    
    if (slide_mode == 'regular'):
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        if (~Line.detected):
            midpoint = np.int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        else:
            y_eval = binary_warped.shape[0]
            leftx_base = Line.avg_left_poly[0]*y_eval**2 + Line.avg_left_poly[1]*y_eval * Line.avg_left_poly[2]
            rightx_base = Line.avg_right_poly[0]*y_eval**2 + Line.avg_right_poly[1]*y_eval * Line.avg_right_poly[2]

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
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
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
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

        #append to line object
        Line.add_points(lefty, leftx, righty, rightx)

    #elif (slide_mode == 'convolution'):
 

def plot(binary_warped, Line):
    #plots and annotates images

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0], dtype=np.int)
    left_fitx = np.array(Line.avg_left_poly[0]*ploty**2 + Line.avg_left_poly[1]*ploty + Line.avg_left_poly[2], np.int)
    right_fitx = np.array(Line.avg_right_poly[0]*ploty**2 + Line.avg_right_poly[1]*ploty + Line.avg_right_poly[2], np.int)

    out_layer = np.zeros_like(binary_warped)
    out_img = np.uint8(np.dstack(( out_layer, out_layer, out_layer)))

    pts_r = np.vstack((right_fitx,ploty)).astype(np.int32).T
    cv2.polylines(out_img,  [pts_r],  False,  (0, 255, 0),  30)

    pts_l = np.vstack((left_fitx,ploty)).astype(np.int32).T
    cv2.polylines(out_img,  [pts_l],  False,  (0, 255, 0),  30)

    pts = np.concatenate((pts_l, np.flipud(pts_r)))
    cv2.fillPoly(out_img,  [pts], (0,0,255))

    return out_img


def process_img(input_img):
    global myLine 

    #perform undistortion
    input_img = undistort(input_img, mtx, dist)

    #apply perspective transform
    per_img = perspective_transform(input_img, source_points, destination_points)

    #perform color and sobel thresholding
    thresh_image, color_binary = threshold(per_img, color_s_thresh, color_l_thresh, sobel_dir_thresh, sobel_mag_thresh)  

    #set parameters for myLine object
    myLine.set_param(input_img.shape, 3/110, 3.7/780)

    #finding and fitting lane lines
    find_lines(thresh_image, myLine)
    
    
    if (myLine.detected):
        #marking lane lines
        wraped_marked_img = plot(thresh_image, myLine)

        #applying inverse perspective transform
        marked_img = perspective_transform(wraped_marked_img, destination_points, source_points)

        #adding marked image to original image
        added_img = cv2.addWeighted(input_img, 1, marked_img, 0.5, 0)

        #annotate image
        annotate_img = cv2.putText(added_img,"Line curveture: {0:.2f} km".format(myLine.radius_of_curvature/1000), (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

    else:
        annotate_img = np.copy(input_img)

    #Adding marked lines to original images
    return annotate_img


def main():
    #global calibration coefficients
    global mtx
    global dist
    global myLine

    #perform calibration
    if calibrate:
        [mtx, dist] = calibrate(calibrate_img_dir, chess_size)\
            
    #image mode
    if (mode == 'image'):

        #read images in directory
        images = os.listdir(img_dir)
        for image in images:
            if (image[-3:]!="jpg"):
                images.remove(image)
        n_images = len(images)

        #iterating images in directory
        for i in range(n_images):
            
            #clear myLine for each image
            myLine = Line();
            #read image
            input_img = mpimg.imread(img_dir + images[i])

            #obtain marked image
            final_img = process_img(input_img)
            
            #save images
            mpimg.imsave((img_out_dir+images[i]), final_img)

            #showing images
            plt.subplot(n_images,2,(2*i+1))
            plt.imshow(input_img)
            plt.subplot(n_images,2,(2*i+2))
            plt.imshow(final_img)

        plt.show()

    elif (mode == 'video'):
        input_clip = VideoFileClip(video_dir)
        output_clip = input_clip.fl_image(process_img)
        output_clip.write_videofile(video_dir_out, audio=False)

 
main()
sys.exit()










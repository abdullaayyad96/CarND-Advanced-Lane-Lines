## Project Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_figures/undistort.jpg "Undistorted"
[image2]: ./test_images/test2.jpg "Road image"
[image3]: ./writeup_figures/undistort_test.jpg "Undistorted test"
[image4]: ./writeup_figures/thresholding.jpg "Thresholding"
[image5]: ./writeup_figures/per_trans.jpg "Perspective Transform"
[image6]: ./writeup_figures/line_find_momentum.jpg "Line finding with momentum"
[image7]: ./writeup_figures/line_find_no_momentum.jpg "Line finding without momentum"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I have considered the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients and utilizing them to undistort an image

The code for the calibration and undistortion are provided in lines 37-72 in the code.py file.  

First step is preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I have assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The detection of chessboard corners have been done using `cv2.findChessboardCorners()` and the size of the chessboard need to be predetermined by the user. I have set the parameters of the undistortion in lines 19-23 of code.py.

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following results: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-correcttion

The first step was applying distortion correction on road images based on the camera matrix and distortion coefficients obtained in the calibration step described above. The same steps for undistortion are applied here yielding the results shown below:

![alt text][image3]

#### 2. Thresholding

In my code, I used a combination of color and gradient thresholds to generate a thresholded image. This step is shown in line 106-183 of the code.py file. For color thresholding, a combination of saturation and lightness thresholding steps are utilized while for gradiant thresholding both gradient magnitude and direction are used. In order to make this step more robust to lighting and road conditions, instead of using fixed cutoff value dynamic thresholds are implemented instead. This is done by specifying a certain region of the image and selecting a certain percentage of pixels which have the highest corresponding value are used. The specified region is a polygon with the following vertices:

| Point        | x-dimension   | y-dimension |
|:-------------:|:-------------:|:-------------:|
| left bottom    | 0.1 * image_width    |   0.93 * image_height     |
| left top       | 0.43 * image_width   |   0.625 * image_height    |
| right top      | 0.57 * image_width   |   0.625 * image_height    |
| right bottom   | 0.9 * image_width    |   0.93 * image_height     |

Here's an example of my output for this step. 

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

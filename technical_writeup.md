## Technical Writeup

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

[//]: # "Image References"

[image1]: ./writeup_figures/undistort.jpg "Undistorted"
[image2]: ./test_images/test2.jpg "Road image"
[image3]: ./writeup_figures/undistort_test.jpg "Undistorted test"
[image4]: ./writeup_figures/thresholding.jpg "Thresholding"
[image5]: ./writeup_figures/per_trans.jpg "Perspective Transform"
[image6]: ./writeup_figures/line_find_momentum.jpg "Line finding with momentum"
[image7]: ./writeup_figures/line_find_no_momentum.jpg "Line finding without momentum"
[image8]: ./writeup_figures/fit_poly_plot.jpg "Plotting fitted polynomial"
[image9]: ./output_images/test2.jpg "Output road image"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I have considered the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients and utilizing them to undistort an image

The code for the calibration in the `calibration.py` file.  

First step is preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I have assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The detection of chessboard corners have been done using `cv2.findChessboardCorners()` and the size of the chessboard need to be predetermined by the user. 

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following results: 

![alt text][image1]

### Pipeline (single images)

The code for the pipeline implementation in the `main.py` and `Line.py`.  

#### 1. Distortion-correction

The first step was applying distortion correction on road images based on the camera matrix and distortion coefficients obtained in the calibration step described above. The same steps for undistortion are applied here yielding the results shown below:

![alt text][image3]

#### 2. Thresholding and Region Masking

In my code, I used a combination of color and gradient thresholds to generate a thresholded image. This step is shown in line 69-146 of the `main.py` file. Since lane lines are often more saturated and bright than the rest of the road, color thresholding using saturation and lightness channels is applied. Additionally, gradient features are used to extract lines, both gradient magnitude and direction of each pixel are compared to a cutoff value to predict whether they are part of lane lines. 

In order to make this step more robust to lighting and road conditions, instead of using fixed cutoff value, dynamic thresholds are implemented instead. This is done by specifying a certain region of the image and selecting a certain percentage of pixels which have the highest corresponding value to be used. The code for region masking can be found in lines 47-66 of `main.py`. The specified region is a polygon with the following vertices:

| Point        | x-dimension   | y-dimension |
|:-------------:|:-------------:|:-------------:|
| left bottom    | 0.1 * image_width    |   0.93 * image_height     |
| left top       | 0.43 * image_width   |   0.625 * image_height    |
| right top      | 0.57 * image_width   |   0.625 * image_height    |
| right bottom   | 0.9 * image_width    |   0.93 * image_height     |

After the region is masked, the color and gradient matrices are scaled and cutoff value is calculated for each matrix based on a preset cutoff percentage, and cutoff is applied accordingly. Afterwards, The output of both steps of thresholding are added together. The motivation behind this is to make the line detection (described in section 4) less responsive to noise and errors as more emphasis will be given to pixels detected by both color and gradient thresholding. 

Here's an example of my output for this step. 

![alt text][image4]

#### 3. Perspective transform 

Perspective transform is applied to obtain a birds-eye view of the road. This view will make other steps along the pipeline easier. My code applies perspective transform in lines 37-44 of `main.py`. The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. These points were hardcoded as shown in lines 19-22 and were chosen manually to be as follows:

| Source        | Destination   |
|:-------------:|:-------------:|
| 594, 450      | 320, 0        |
| 685, 450      | 960, 0        |
| 280, 675      | 320, 720      |
| 1040, 675     | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Results of applying perspective transform on a test image and the images obtained from previous steps are shown below:

![alt text][image5]


#### 4. Lane-Line Identification

Lane-Line identification is applied in line 148-375 of `main.py`. This method accepts a thresholded image and a Line object (Implementation of Line class can be seen in `Line.py`), where it detects pixels that are part of a line using a sliding window approach and pass these pixels to the Line object for processing and polynomial fitting. It also returns an image highlighting the sliding widows and the detected pixels. The most important step of this process is determining the positions of the searching windows. I have implemented two methods of sliding windows in my code:

##### 4.1 Regular sliding window

The first step is adding all pixels in the combined thresholded image along each vertical column in the lower third of the image. The peaks of this summation on the right and left side will determine the bases of the lane lines and will be the initial position of the searching window. In my implementation, I ignore vertical columns near the edges as the probability of lane lines to appear in edges is low. After determining the positions of the base windows, pixels within that window are appended to a list containing the points that are part of the lane lines. Additionally, the averaged x-coordinates of these points will determine the position of the next window and so forth. This approach is applied in lines 182-272 of `main.py`

##### 4.2 Convolutional sliding window

This is very similar to the first approach but rather than taking peaks of summed vertical columns or averaging x-positions to determine the center of the next level windows, vertical columns are summed and the resulting histogram is convolved with a window template and the peaks of the convolved signals become the center of the next level window. This approach is applied in lines 274-382 of `main.py`.

The results obtained from implementing both methods are very similar.

In addition to determining the position of windows utilizing the above two methods, the concept of shifting momentum is utilized where it is assumed that if windows are shifting in a certain direction, it is more likely that line pixels will continue to progress in the same direction. Thus in addition of determining next level window positions using the previously described approaches, previous shifts of window positions can help determine future shifts. This is very advantageous with dashed curved lines where there might be segments/windows with no detected line pixels which would severely affect the accuracy of lane detection as the window shifts can be irrelevant to curvature of the line. The below pictures show the advantage of utilizing this concept:

![alt text][image7]
![alt text][image6]

Finally, once pixels of right and left lane lines have been determined, they are passed to the Line object in `Line.py` for processing as shown in the below sections.

#### 5. Processing lines

As mentioned earlier, line points are passed to a Line object to be processed. The implementation of the Line class can be seen in `Line.py`. The first step is fitting the newly detected points into two second degree polynomials, one for the left and the other for the right line. 

Upon obtaining fitted polynomials, several tests are undertaken to ensure the validity of these newly obtained polynomials as shown in the `sanity_check` function in lines 110-142 of `Line.py`. First test is calculating the average distance between the left and right lines and comparing it to the nominal distance of 3.7m between lane lines as indicated by the U.S. regulations. This average distance is calculated by integrating the distance between left and right polynomials over a specified range and then dividing the result by the width of this range as shown in the `calc_avg_distance` function in lines 97-108 of `Line.py`. And by comparing this value to the nominal distance, an error value is obtained. The second validity test is similar to the first one but instead of comparing the distance between newly obtained lines to nominal values, they are compared to the averaged distance of averaged lines obtained from previous lines (if previous lines were detected of course). The final validaty test compared the distance between line at the base of the frame and also compares it to the nominal value of 3.7m. The reason this test is needed in addition to the first one is because there have been observed cases where the lines are very close at one end but two far or the other end, yet the average distance is similar to the nominal case. However, by also checking the distance one end of the image, the probability of such cases occurring  is greatly reduced. 

If the newly fitted polynomials are deemed valid, they are averaged with previously obtained polynomials in order to smoothen changes in lane lines and to make the pipeline less susceptible to errors.

#### 6. Radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of a second degree polynomial in the form of `A*y^2 + B*y + C` can be calculated using: `R = ((1 + (2*A*y + B)^2)^1.5) / (2*A)`. Prior to proceeding with calculating curvatures, the polynomials are first converted from pixel dimensions to actual dimensions as shown in the `cvrt_2_act` function in `Line.py`. This is done using conversion factors which were set manually. Afterwards, the radius of curvature is calculated separately for left and right lines and then averaged to get the curvature of lane. In case the curvature of the lane is very large and exceeds a certain limit (4km in my case), the lane will be considered a straight line with a curvature radius of infinity. This is implemented in the `find_curvature` functions in line 64-85 of `Line.py`.

As for calculating the position of the vehicle with respect to the center of the lane, my code implements this in the `calc_displacement` function in `Line.py`. Initially, the x-position of the midpoints between the right and left lines is calculated at the base of the image and this value is compared to the center of the image as the camera is assumed to be installed in the center of the vehicle. Once the difference is calculated in pixel dimensions it is converted to actual dimensions using a manually set conversion factor. In this case, positive values indicate the vehicle is to the left of the lane center while negative values indicate the opposite. 

#### 7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After detecting the lane lines as described in the steps above, the detected lines are plotted on the original image. First, the fitted polynomial are drawn according to the perspective transformed image as implemented in the `plot` function in `code.py` using `cv2.polylines` and `cv2.fillpoly` resulting in an images as shown below:

![alt text][image8]

Finally, inverse perspective transform is applied and the resulting image is added to the original input image and annotated resulting in the following output image:

![alt text][image9]

---

### Testing pipeline on video

Here's a [link to my video result](./output_videos/project_video.mp4).

![test_video](output_videos\project_video_Trim.gif)

---

### Discussion

The main flows of my pipeline were observed when testing it on some more challenging videos, where no matter how I tuned my thresholding steps and percentages I could not get a combination which can clearly detect lane lines while neglecting shadows and fake lines in the image. Thus better techniques of thresholding could be researched. One possible idea is changing the thresholding parameters depending on the coordinates of the image since different coordinates will have different pixel densities.

Additionally, the Line detection process can be made smarter where information obtained from detecting line points from both lines are mutually utilized in order to determine the best combination of shifting left and right windows simultaneously in the sliding window approach. This can be advantageous since usually the detection of one side is more reliable than the other either due to lighting/shadow conditions or simply because one side is dashed while the other is continuous. Also a dynamic window size can be advantageous as the pixel density varies as we move further from the vehicle. 

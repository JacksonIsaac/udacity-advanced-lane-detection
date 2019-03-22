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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./advanced_lane.ipynb". There are two functions to do the camera calibration.

First function contains the code to calculate the object points and the image points from the given set of sample Chessboard images.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

I will be using the 'straight_lines1.jpg' sample file to demostrate the different steps involved in the pipeline.

#### 1. Provide an example of a distortion-corrected image.

I have defined the distortion-correction code in a function called `cal_undistort()`.

First we get the calibration object and image points by using the Chessboard images provided. After that I got the correction matrix using cv2.calibrateCamera() function, which we pass it to the cv2.undistort() function which applies the transformation matrix on the input image and returns the corrected image.

Sample output is given as below:

![Original Image][./test_images/straight_lines1.jpg]
![Camera Calibrated Image][./output_images/straight_lines1_undistort.jpg]

Output for other sample images are stored in 'output_images' folder with '_undistort' suffix.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code is defined in a function called `custom_thresh()`. Here's an example of my output for this step. 

![Color Transformed Image][./output_images/straight_lines1_mag_gradient.jpg]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `image_warp()`.  The `image_warp()` function takes as inputs an image. It calls the `cal_undistort()` which distort-corrects the input image, and using hardcoded src and destination points, which I calculated manually from [link](https://yangcha.github.io/iview/iview.html). I also referred [knowledge hub](https://knowledge.udacity.com/questions/30221) for ideas.

I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([
    [490, 480],
    [810, 480],
    [1250, 720],
    [40, 720]
])
dst = np.float32([
    [0, 0],
    [1280, 0],
    [1250, 720],
    [40, 720]
])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective Transformed Image][./output_images/straight_lines1_perspective_trasformed.jpg]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

From the 'Advanced Computer Vision', I referred the code to try out different ways to detect the lanes. I tried running the different methods like using histogram, sliding window given in the lesson on given input images and some of the samples are as below:

![Lane detection using Windowing method][./output_images/straight_lines2_lane_window.jpg]
![Lane detection using polynomial function][./output_images/straight_lines2_lane_roi.jpg]

After lot of trial and error, I finalized the lane detection pipeline which is defined in a function called `draw_lanes()`. Here I use windowing, then take the histogram of left half and right half of the image to distinguish between left and right lanes. After that I use combination of polynomial functions and polyfit for top and bottom of the lane image so that the polygon is completely fit on the real image properly. I faced some issues where lanes were broken, lanes were further away, etc. hence filled in the polygon from top to bottom using partial lane lines.

Final output looked as below (with lane drawn on the real image, along with Radius of curvature and Car position from center):

![Lane detection using combination of polynomial functions][./output_images/straight_lines2_lane_final.jpg]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature (RoC) of the lane and the position of the vehicle with respect to center.

I referred to the 'radius of curvature' lesson for calculation. The basic idea is to fit in a polynomial function, using which we can calculate RoC using the formula:

R = ((1+(2Ay+B)^2)^1.5) / |2A|,

where A and B are the first and second order derivate of the polynomial function (of degree 2), respectively.

![Lane detection with Radius of Curvature][./output_images/straight_lines2_lane_final.jpg]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the plotting of detected lane (from perspective transformed image) onto real image in the function `draw_lanes()`.  Here is an example of my result on a test image:

![Lane detection mapped on Real Image][./output_images/straight_lines2_lane_final.jpg]

To map the lines from perspective image back to the real image, we use the inverse matrix of perspective transformation by passing (destionation, source) points to the getPerspectiveTransform() function.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I tried different thresholding methods like binary, red-blue-green, HLS, LAB (reference)[https://medium.com/@ajsmilutin/advanced-lane-finding-5d0be4072514] color transformations. I found out the S channel of HLS, Red channel and B channel of LAB works really good and combination of these will help us detecting the lane clearly. Although, I faced issues with using Red channel in some of the sample images, hence decided to use combination of S and B channels of HLS and LAB, respectively.

S Channel is able to detect lanes really well even in case of shadow on road. And B channel of LAB was able to detect the yellow lanes really well.

Some of the limitations of the current pipeline is that it is not able to handle the curves properly, broken lanes or lanes having more gap between simultaneous lines.

To overcome this, I could probably try to change the area that I use to detect lanes or have less window (window height will be more, so more area to find lanes with more gap between them). For curves, I might need to find another equation or way to handle them, current polynomial function along with using only portion of image to try and detect the lanes are not sufficient enough.

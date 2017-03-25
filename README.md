# Self Driving Car Nano Degree
---
## Vehicle Detection Project (P5)

The primary goals of this project are detailed below:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Apply any color transformations, binned color features, histograms of color to HOG feature vector.
* Normalize features and randomize selection of data for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in individual images.
* Execute image processing pipeline on a video stream using the sample videos provided in the previous Advanced Lane Lines project and create a heat map of recurring detections in each frame to reject outliers while following detected vehicles.
* Estimate bounding boxes for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/pipeline_output.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells#12 and cells#13 of the IPython notebook P5-Vehicle-Detection-ProjectV1.ipynb. The methods extract_features() and extract_features_all() defined are reusable methods in cell#6 are used to extract HOG features. To extract HOG features, I have used 8792 Vehicle and 8968 Non Vehicle images. The extract_features() function focuses on only getting HOG features and extract_features_all() tries to extract Spatial Binning and Color Histograms.

Random images of vehicle and non-vehicle images are provided below:

![alt text][image1]

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  Displayed random images each of the two classes to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

After experimenting and some trail&error with different parameters, two different types of HOG extractions were found to improve accuracy of the classifier dramatically:
* Approach#1. HOG feature extraction with following parameters:
  - colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
  - orient = 11
  - pix_per_cell = 16
  - cell_per_block = 2
  - hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

* Approach#2. HOG feature extraction with following parameters:
    - colorspace = 'YCrCb'
    - orientations = 9
    - pix_per_cell = (8, 8)
    - cell_per_block = (1, 1)
    - hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

Statistics captured while training SVC classifier using Approach#1 are given below:
```
HOG Feature Extraction execution in:  59.19  seconds
Training execution time(including split):  1.8  seconds
Test Accuracy of SVC =  0.984
Prediction of SVC:  [ 1.  1.  1.  1.  0.  1.  0.  0.  0.  1.]
For  10 labels:  [ 1.  1.  1.  1.  1.  1.  0.  0.  0.  1.]
It took 0.00209  seconds to predict 10  labels with SVC
```

Statistics captured while training SVC classifier using Approach#2 are given below:
```
Full Feature Extraction execution in:  64.52  seconds
Training execution time(including split):  15.17  seconds
Test Accuracy of SVC =  0.9947
Prediction of SVC:  [ 1.  1.  0.  0.  0.  0.  0.  0.  1.  0.]
For  10 labels:  [ 1.  1.  0.  0.  0.  0.  0.  0.  1.  0.]
It took 0.00191  seconds to predict 10  labels with SVC
```
I chose Approach#2 as the accuracy has been found to be around 99.5%.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
Two approaches were tried during this project to train an SVC (Support Vector Classifier). A reusable method train_classifier_svc(car_features, notcar_features, feature_set='ALL') written in code cell#11 has been used to achieve both objectives.

* Approach#1: The SVC classifier was trained using HOG feature extraction only. The reusable method extract_features() method has been used to perform feature extraction prior to training SVC, by using the following parameters:
```
colorspace = 'YUV'
orientations = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
```
A hog channel of "ALL" ensures all channels are selected in a color space. While training SVC using HOG channel only the colorspaces "YUV" and 'YCrCb' produced better accuracy. Details of training SVC using this approach show accuracy of 98.4%. The source code for this training can be found in code cell#12.

* Approach#2: The SVC classifier was trained using HOG feature, Spatial Binning and Color Histogram features stacked. The reusable method extract_features_all() sklearn.hog() method, by using the following parameters:
```
colorspace = 'YCrCb'
orientations = 9
pix_per_cell = 8
cell_per_block = 1
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
A hog channel of "ALL" ensures all channels are selected in a color space. The reported test accuracy score was 99.5%.The source code for this training can be found in code cell#13. The Spatial size and hist_bins parameters are similar to what was used in lessons.

I chose SVC trained from Approach#2 to proceed further in testing.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
After trying different values for scaling and y-coordinate start and stop parameters, the following parameters were chosen to be optimal.
ystart = 336
ystop = 656
scale = 1.5
These parameters confine the viewing area that can be imagined as road level with viewing angle as straight through front wind shield. Setting ystart=336 ensures the search area does not go above to skyline or tree line for other vehicles. The scaling value between 1 to 1.5 has been found to be optimal. Anything less than scale value of 1 has been found to produce inaccurate results. The source code can be found in IPython notebook cell#43.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
Finally I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result with negligible to less false positive results.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video. The examples comparing original image, heatmap, thresholded heat map and labeled heatmap images samples provided below:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]
---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall my pipeline worked really well batched across frames. However I could see a flicker while cars enter and exit frames. Since the accuracy of classifier is around 99.5%, we could expect few inaccurate predictions in window frames. The pipeline could probably fail in scenarios where the new vehicles that are not similar to training dataset appear in the video. The test videos used are confined to freeway samples. Vehicles coming in opposite direction or objects that appears to be like vehicles or vehicles far away which could above ystart point might not be detected with bound boxes drawn. Another issue that could not be easily resolved and would require more trail and error is to guess the best overlap window. Heavy traffic conditions could also create overlap labels when the chances of vehicles closer to each other is high. However there is scope for improvement with more research towards the following areas:
* Identify distance and speed of the vehicles to predict labels in subsequent frame to avoid edge case scenarios.
* Using Neural Networks to ensure better search and predict capabilities.

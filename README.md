## Vehicle Detection Pipeline

This is a discussion for the "vehicle detection" exercise in Udacity's Self Driving Car Program.
Images to train the later introduced classifier were provided by the instructors as well as test videos, etc.

[//]: # (Image References)
[hog_car]: ./hog_car_yuv_allChannels.png
[hog_noncar]: ./hog_noncar_yuv_allChannels.png
[grids]: ./grids.png
[heat1]: ./heat_test.png
[heat2]: ./heat_test_after.png
[video1]: ./result_video.mp4

*code*:
  * vd.py
  * image_functions.py

*run*:
  * **python vd.py --use_test_images 1 --do_train 1 --color_space YUV**

   .. to train a linear classifier on a given set of test images and to run the pipeline on a set of images. all within the YUV colorspace. (the trained classifier is stored in a pickle file)

  * **python vd.py --use_test_images 0 --do_train 0 --color_space YUV**

   .. read in a pickled trained linear classifier for the YUV color model and run the pipeline on a video stream.


---

**Vehicle Detection Pipeline Preparation**

Before we can run the pipeline we have to train a classifier, which helps us to classify car and not car images.
For this project we chose the *Linear Support Vector Classification* from sklearn over other techniques like Decision trees, etc. Reason is that this class has been proven to work well with HOG feature extraction - which is one of the fundamental tools we use to extract car image features.

In the function "train" we train a LinearSVC classifier on a training set and compute its accuracy (to determine if we over- or under-fit) on a test set.
In order to derive our train and test data we compute the normalized values of image feature from a given data set (comprised of car/not-car images). Note, that the train/test data is split randomly using sklearn' train_test_split function in the "get_train_AND_test_data" function.

The features we extract from images are its HOG features on ALL channels, plus spatial & histogram features.
The actual values to control these feature extraction algorithms are discussed below.

In addition we precomputed a set of windows, which will later be used for a sliding window approach to run each window through the trained classifier's predict function and tell if the subimage depicts a car or not.

Note, that a) we ignore the upper half of the image and b) box sizes increase with decreasing distance to our car.

The various sizes of sliding windows are defined in the function "compute_windows".

The following picture shows the computed windows on an example (720/1280) image:
![alt text][grids]


<br><br>
**Vehicle Detection Pipeline Steps**

The central function in the project (in vd.py) is "vd_pipeline", which represents the essential steps to detect cars within an image using a trained linear classifier.

The steps to detect cars are:

* read in constant parameters (for controlling hog feature detection, etc.)
* for all the (various sized) windows run a sliding window approach:
  - resize "window image" to the size of our train images (64,64)
  - extract the feature vector based on the given params (hog features, spatial binning, etc.)
  - normalize the extracted feature vector similar to how the training set features were normalized to avoid certain features dominating others due to their numerical values
  - and then run a prediction using our (pickled) classifier on the list of feature vectors (list size is size of sliding windows), which returns a list of predictions (0 and 1 values. 0 .. no car / 1 .. car)
* for the windows, which encapsulate predicted cars (= hot_windows), we compute a heatmap. The motivation for a heatmap is to detect recurring detections frame by frame to reject outliers and follow detected vehicles. In addition using a heatmap approach is also useful to meld various windows detecting the same car into one encapsulating bounding box.
* this heatmap is added to a fixed sized (size=10) collection (deque) of heatmaps. taking the sum gives us the heatmap of the last 10 video frames
* use scipy.ndimage.measurements' label functionality to compute the number of "islands" (i.e. cars) in a thresholded heatmap
* finally, draw boxes around the detected car labels

Chosen parameter configuration are discussed below.


<br><br>
**Feature Extraction incl. HOG features**

Here we give a quick visualization of our HOG feature extraction.
The upper left is the original gray scale image and the other images denote the hog features for the 3 YUV channels.
The 3 hog pics show the dominant gradient direction within each cell with brightness corresponding to the strength of gradients in that cell.

![alt text][hog_car]

Below is an example with extracted hog features on a non-car image.

![alt text][hog_noncar]

*Take-away:* It seems that only the first channel really reflects contours of a car correctly and other channels will not impact much.
However in the next section we see that enabling "ALL" channels for the feature extraction increases our accuracy on the test set by a bit.
And more importantly we have seen in experiments on resulting video outputs that turning "ALL" channels on produces better results (less false positives).

Hog features are extracted in the code in the "get_hog_features" function in "image_functions.py".


<br><br>
**Parameters**

As mentioned above in the "train" section we extract HOG features, plus spatial & histogram features from images when we train our classifier as well as when we predict if a window shows a car or not.

Below is a quick study of various parameters (color spaces, etc.).

Parameter Exploration:

| color_space | orient.| pix_per_cell | cell_per_block | hog_channel |test-accuracy| cars missed | false pos. | comments (about time, etc.)|
|-------------|:------:|-------------:|---------------:|------------:|------------:|------------:|-----------:|---------------------------:|
|     HSV     |    9   |      8       |        2       |      0      |   0.9582    |     0       |     >50    |  (fast to train: 8.98sec)
|     HSV     |    9   |      8       |        2       |      2      |   0.9787    |     0       |     >50    |  (fast to train: 7.5sec)
|     HSV     |    9   |      8       |        2       |    "ALL"    |   0.9903    |     1       |      4     |  -
|     HSV     |    9   |      8       |        2       |    "ALL"    |   0.9903    |     1       |      4     |  -
|     YUV     |    9   |      8       |        2       |    "ALL"    |   0.9886    |     0       |      1     |  makes NaN
|     YUV     |    9   |      8       |        2       |    "ALL"    |   0.9929    |     0       |      1     |  -
|     YUV     |    9   |      8       |        2       |      0      |   0.9747    |     0       |      5     |  8.05sec, makes NaN, 7.7sec
|     YCrCb   |    9   |      8       |        2       |    "ALL"    |   0.9895    |     0       |      0     |  (16.87 sec)
|     YUV     |    9   |      16      |        2       |    "ALL"    |   0.9872    |     2       |      3     |  makes NaN
|     YUV     |    9   |      8       |        2       |    "ALL"    |   0.9844    |     0       |      5     |  no spatial and histo
|     YUV     |    9   |      8       |        2       |    "ALL"    |   0.9918    |     0       |      2     |  5.84 sec for car boxing
|     YUV     |    9   |      8       |        2       |    "ALL"    |   0.9906    |     0       |      2     |  5.8 sec for car boxing
|     YCrCb   |    9   |      8       |        2       |    "ALL"    |   0.9864    |     0       |      0     |  6.06 for car bb

It turned out the "YUV" color space gave good accuracy on the test set. Also as discussed above computing hog features on all channels gave slightly better results and was worth the extra computational overhead.
Further, some other parameters were used for the hog extraction. And the histogram and spatial feature extraction are enabled in my program.

These are the params used - found in the function "get_params"

* color_space = YUV .. *color space*
* orient = 9  .. *HOG orientations*
* pix_per_cell = 8 .. *HOG pixels per cell*
* cell_per_block = 2 .. *HOG cells per block*
* hog_channel = "ALL" .. *Can be 0, 1, 2, or "ALL"*
* spatial_size = (16, 16) .. *Spatial binning dimensions*
* hist_bins = 16 .. *Number of histogram bins*
* spatial_feat = True .. *Spatial features on or off*
* hist_feat = True .. *Histogram features on or off*
* hog_feat = True .. *HOG features on or off*


<br><br>
**Heatmap**

This section is to briefly visually demonstrate results of heatmap approach discussed above to a) reduce the number of false positives and b) to compute the enclosing bounding box out of nested bounding boxes.

This cluster of consecutive video frames (each frame ran through the vehicle detection pipeline without the use of heatmaps) shows the problem a) and b) mentioned before: a few false positive detections and nested bounding boxes.
![alt text][heat1]

Once we turn on the "heatmap functionality" we see less false positives and only one enclosing bounding box per car.
![alt text][heat2]


<br><br>
**Result video**

Here's a [link to my video result](./result_video.mp4)


<br><br>
**Challenges / Improvements / Tips**

* A challenge was to resolve some "NaN" issues in the feature vector when using YUV or YCrCb color schemes. Due to lack of time I did not hunt down how they occur and just filter these values out (since I did not want to mess with the position of my values in feature vectors with the risk of getting wrong prediction results, I simply overwrote NaN values with 0.)

* Other challenges are related to efficiency: in order to speed up the implementation we only run the engine on the lower half of the image, we precomputed the sliding windows and we executed the prediction step for all sliding windows at once: i.e., the input to predict in the "search_windows" function is a list of feature vectors; and returns a list of predictions.
More optimizations can definitely be done like reducing the number of features or precomputing hog features only once for the entire image.

* *Problems* will definitely occur in trickier conditions like darkness, rain, etc. and I would assume an approach using (Lidar) distance sensors sounds more appropriate to detect neighboring cars.
In addition we did not train here for other obstacles like pedestrians, cyclists, road hazards, etc.
Also there are still some holes in the video stream where I did not detect the car or were false positives are detected, which could result in unwanted braking or lane changing.

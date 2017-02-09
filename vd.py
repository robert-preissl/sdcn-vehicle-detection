
from PIL import Image

import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle
import tensorflow as tf
tf.python.control_flow_ops = tf # for https://github.com/fchollet/keras/issues/3857

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio

import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

from image_functions import *
import collections


# some runtime parameters to be set via command line
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('use_test_images', '1', "Do a camera calibration. if not, read pickled calibration matrix.")
flags.DEFINE_integer('sample_size', '0', "only use a certain number of car and not car images for training.")
flags.DEFINE_integer('do_train', '0', "read in pickled trained data.")
flags.DEFINE_string('color_space', 'YUV', "color space to use.")

# Usage:
#  - python vd.py --use_test_images 1 --do_train 1 --color_space YUV
#  - python vd.py --use_test_images 0 --do_train 0 --color_space YUV



# read in training data
# return car and notcar images
def read_train_images():
    # Read in cars and notcars
    cars_gti_far         = glob.glob("./vehicles/GTI_Far/image*.png")
    cars_gti_left        = glob.glob("./vehicles/GTI_Left/image*.png")
    cars_gti_middleclose = glob.glob("./vehicles/GTI_MiddleClose/image*.png")
    cars_gti_right       = glob.glob("./vehicles/GTI_Right/image*.png")
    cars_kitti           = glob.glob("./vehicles/KITTI_extracted/*.png")

    cars = cars_gti_far + cars_gti_left + cars_gti_middleclose + cars_gti_right + cars_kitti
    print("car images. images len = ", len(cars))

    non_cars_extra       = glob.glob("./non-vehicles/Extras/*.png")
    non_cars_gti         = glob.glob("./non-vehicles/GTI/*.png")

    notcars = non_cars_extra + non_cars_gti
    print("non-car images. images len = ",len(notcars))
    sample_size = min(len(cars), len(notcars)) # make sure we have the same number of car/non-car pics

    # Reduce the sample size for quicker debugging if flag set
    if FLAGS.sample_size > 0:
        print("use sample_size = ", sample_size)
        return cars[0:FLAGS.sample_size], notcars[0:FLAGS.sample_size]
    else:
        print("use sample_size = ", FLAGS.sample_size)
        return cars[0:sample_size], notcars[0:sample_size]


# compute features of car and not-car images (typically done during training)
def compute_features(cars, notcars):
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size,\
            hist_bins, spatial_feat, hist_feat, hog_feat = get_params()

    print('Using:',orient,'orientations',pix_per_cell,
          'pixels per cell and', cell_per_block,'cells per block')

    car_features = extract_features(cars, read_img=True, color_space=color_space,
                          spatial_size=spatial_size, hist_bins=hist_bins,
                          orient=orient, pix_per_cell=pix_per_cell,
                          cell_per_block=cell_per_block,
                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                          hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, read_img=True, color_space=color_space,
                          spatial_size=spatial_size, hist_bins=hist_bins,
                          orient=orient, pix_per_cell=pix_per_cell,
                          cell_per_block=cell_per_block,
                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                          hist_feat=hist_feat, hog_feat=hog_feat)

    return car_features, notcar_features


# normalize car and not-car features
def normalize(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    return scaled_X, X_scaler


# compute train and test data using doing a random split between the two
def get_train_AND_test_data(car_features, notcar_features, scaled_X):
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    return X_train, X_test, y_train, y_test


# do the training and do the testing (i.e., compute accuracy on test set to determine if we over/under-fit).
# return: a trained classifier
def train(X_train, X_test, y_train, y_test):
    # Use a linear SVC
    svc = LinearSVC() # TODO: try SVM, etc.

    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc


# Parameter Experiments:

#  color_space | orient | pix_per_cell | cell_per_block | hog_channel || accuracy | cars missed | false pos. | comments (about time, etc.)
#  ------------|--------|--------------|----------------|-------------||----------|-------------|-------------
#      HSV         9           8               2                0          0.9582      0             >50      (fast to train: 8.98sec)
#      HSV         9           8               2                2          0.9787      0             >50      (fast to train: 7.5sec)
#      HSV         9           8               2               "ALL"       0.9903      1             4
#                                                                          0.9903      1
#      YUV         9           8               2               "ALL"       0.9886      0             1    (fast training time. 3 sec), makes NaN
#                                                                          0.9929
#      YUV         9           8               2                 0         0.9747      0             5       8.05sec, makes NaN, 7.7sec
#      YCrCb       9           8               2               "ALL"       0.9895      0             0     (16.87 sec)
#      YUV         9           16              2               "ALL"       0.9872      2             3    (fast training time. 3 sec), makes NaN
#      YUV         9           8               2               "ALL"       0.9844      0             5    no spatial and histo
#      YUV         9           8               2               "ALL"       0.9918      0             2    5.84 sec for car boxing
#      YUV         9           8               2               "ALL"       0.9906      0             2    5.8 sec for car boxing
#      YCrCb       9           8               2               "ALL"       0.9864      0             0    6.06 for car bb


# helper function to return the parameters used for vehicle detection
def get_params():
    color_space = FLAGS.color_space # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    return color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size,\
           hist_bins, spatial_feat, hist_feat, hog_feat


# compute all the various sized windows for our image:
#  - ignore upper half of the image
#  - provide various sized windows: smaller ones on top for smaller / further away cars. increase sizes with increasing y
# return: list of windows
def compute_windows(img):
    #                        wdw-x / wdw-y   y-st / y-end       x-st / x-end  x-ovlp / y-ovlp
    xy_wdws_and_start_pos = [[48,    48,      380, 480,          530, 1000,     0.5,    0.5  ],\
                             [96,    96,      380, 570,          390, 1280,     0.5,    0.5  ],\
                             [144,   144,     380, 600,          272, 1280,     0.5,    0.5  ],\
                             [288,   288,     370, 660,          128, 1280,     0.5,    0.5  ]
                             ]

    all_windows = []
    for xy_wdws_and_start_p in xy_wdws_and_start_pos:
        # compute all the little windows we can lay into the image
        windows = slide_window(img,
                           x_start_stop=[xy_wdws_and_start_p[4],xy_wdws_and_start_p[5]],
                           y_start_stop=[xy_wdws_and_start_p[2],xy_wdws_and_start_p[3]],
                           xy_window   =[xy_wdws_and_start_p[0],xy_wdws_and_start_p[1]],
                           xy_overlap  =(xy_wdws_and_start_p[6],xy_wdws_and_start_p[7]))

        # print("windows = ", windows)
        all_windows.append(windows)

    windows = [item for sublist in all_windows for item in sublist] # flatten list of lists .. http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    return windows


# The vehicle detection pipeline:
#  - read in constant parameters
#  - search through all the (various sized) windows:
#     - resize "window image" to the size of our train images (64,64)
#     - extract the feature vector based on the params (hog features, spatial binning, etc.)
#     - and then run prediction on a list of feature vectors, which returns a list of predictions (0 and 1 values. 0 .. no car / 1 .. car)
#  - for the windows which encapsulate predicted cars (= hot_windows), we compute a heatmap
#  - this heatmap is added to a fixed sized (size=10) collection (deque) of heatmaps. taking the sum gives us the heatmap of the last 10 video frames
#  - use scipy.ndimage.measurements' label functionality to compute the number of "islands" (i.e. cars) in a thresholded heatmap
#  - finally, draw boxes around the detected car labels
def vd_pipeline(img, draw_image, classifier, X_scaler, windows, heatmaps):
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size,\
        hist_bins, spatial_feat, hist_feat, hog_feat = get_params()

    # use the windows from above and search in all these windows for cars
    hot_windows = search_windows(img, windows, classifier, X_scaler, color_space=color_space,
                    spatial_size=spatial_size, hist_bins=hist_bins,
                    orient=orient, pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block,
                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                    hist_feat=hist_feat, hog_feat=hog_feat)

    heatmap = get_heatmap_hot_windows(hot_windows)
    heatmaps.append(heatmap)
    heatmaps_sum = sum(heatmaps)

    # Now apply thresholding and labeling to hm
    heatmap_labels = label_heat_map_islands(heatmaps_sum, threshold = 8)
    heatmap_window_img = draw_labeled_bboxes(draw_image, heatmap_labels, thick = 3)

    # draw the windows, which encapsulate cars, as boxes on the orig. image
    #window_img = draw_boxes(draw_image, hot_windows, thick=3)

    return heatmap_window_img


# train a classifier from scratch (and save to disk) or read a pickled one
def pickle_train_data():

    classifier = None
    X_scaler = None

    pickle_file_name = "train_data" + str(FLAGS.color_space) + "_CH0.p"

    if FLAGS.do_train:
        # get the training data files
        cars, notcars = read_train_images()

        # compute feature vectors for cars and not-cars
        car_features, notcar_features = compute_features(cars, notcars)

        # filter out NaN values (occur with YUV)
        filter_nan(car_features)
        filter_nan(notcar_features)

        # normalize the data
        scaled_X, X_scaler = normalize(car_features, notcar_features)

        # split the normalized data set into a train & test set
        X_train, X_test, y_train, y_test = get_train_AND_test_data(car_features, notcar_features, scaled_X)
        print('Feature vector length:', len(X_train[0]), ' / shape of X_train = ', X_train.shape, ' / shape of X_test = ', X_test.shape)

        # train a classifier and compute its accuracy (on the test set)
        classifier = train(X_train, X_test, y_train, y_test)

        # save as pickle file
        train_data = { "classifier": classifier, "X_scaler": X_scaler }
        pickle.dump( train_data, open( pickle_file_name, "wb" ) )

    else:
        train_data_pickle = pickle.load( open( pickle_file_name, "rb" ) )
        classifier = train_data_pickle["classifier"]
        X_scaler = train_data_pickle["X_scaler"]

    return classifier, X_scaler


# the main entry to the vehicle detection program
def main():
    print("Start")

    # ----------  for training  -------------
    t = time.time()
    classifier, X_scaler = pickle_train_data()
    t2 = time.time()
    print(round(t2-t, 2), 'Overall prep time: reading data, training, normalizing')
    # ------------------------------------------

    a_test_img = mpimg.imread("./test_images_video_heat/test1.png")
    windows = compute_windows(a_test_img)
    heatmaps = collections.deque(maxlen=10)

    # Check the prediction time for a single sample/or video stream
    t=time.time()

    # we can run the advanced-lane-finding pipeline on a test-image or on a video stream
    if FLAGS.use_test_images == 1:
        #test_imgs = glob.glob("./CarND-Vehicle-Detection/test_images/test*png")
        #test_imgs = glob.glob("./test_images_video/test*png")
        test_imgs = glob.glob("./test_images_video_heat/test*png")
        print("Start with test-images through pipeline")

        window_imgs = []
        only_one_img = True
        for test_img_file in test_imgs:

            if only_one_img:

                test_img = mpimg.imread(test_img_file) # read as RGB
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGRA2BGR) # to get rid of 4th channel
                #print("test_img = ", test_img)
                draw_image = np.copy(test_img)

                # run the vehicle detection pipeline
                window_img = vd_pipeline(test_img, draw_image, classifier, X_scaler, windows, heatmaps)

                # my_plot(window_img)
                window_imgs.append(window_img)

                #only_one_img = False

        print("Done. window_imgs = ", len(window_imgs))
        t2 = time.time()

        #combo_plot_test_imgs(window_imgs)
        #combo_plot_test_9_imgs(window_imgs)
        combo_plot_test_15_imgs(window_imgs)
        #my_plot( window_imgs[0] )


    else:
        print("Run video stream through pipeline")

        # Initialize the video reader, and the writer to write the mp4
        reader = imageio.get_reader('./CarND-Vehicle-Detection/project_video.mp4')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('output_t8_CH0.mp4',fourcc, 20.0, (1280,720))

        t2 = time.time()
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer('./project_video_result.mp4', fps=fps)
        cnt = 0

        for im in reader:
            if cnt % 1 == 0: # for testing we can just check 10% or smaller of the frames of the video.
                rgb_image_scaled = im.astype(np.float32)/255
                # print("rgb_image_scaled = ", rgb_image_scaled)
                draw_image = np.copy(rgb_image_scaled)

                # run the vehicle detection pipeline
                window_img = vd_pipeline(rgb_image_scaled, draw_image, classifier, X_scaler, windows, heatmaps)

                #my_plot(window_img)
                print("window_img = ", window_img.shape)

                b,g,r = cv2.split(window_img)
                new_image2 = cv2.merge([r,g,b])
                processed = (new_image2 * 255.0).astype('u1')

                out.write(processed)

            cnt = cnt + 1

        out.release()
        cv2.destroyAllWindows()
        print("Done: processed ", cnt, " many frames.")

    print(round(t2-t, 2), 'Seconds to compute cars boxes...')


if __name__ == "__main__":
    main()

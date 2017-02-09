
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

# Many functions in here are helper function to plot, to extract features, to compute
#  sliding windows etc.. and are taken from the Udacity class.


# filter out NaN values from an array
def filter_nan(Arr):
    #only for YUV, LUV (having around 10K NaN)
    i = 0
    j = 0
    for b in Arr:
        j = 0
        for a in b:
            if np.isnan(a):
                # print("NAN")
                Arr[i][j] = 0.0
            if np.isinf(a):
                #print("INF")
                Arr[i][j] = 0.0
            j = j + 1
        i = i + 1


# Function to return HOG features and do visualization (if specified)
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Function to extract features from a list of images
# this function calls bin_spatial() and color_hist()
def extract_features(imgs, read_img=True, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = None
        if read_img:
            image = mpimg.imread(file)
        else:
            image = file

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# compute the x start/stop values for our sliding windows
def get_x_start_stop(img, x_start_stop):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]

    return x_start_stop


# compute the y start/stop values for our sliding windows
def get_y_start_stop(img, y_start_stop):
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    return y_start_stop


# Function that takes an image, start and stop positions in both x and y,
#  window size (x and y dimensions), and overlap fraction (for both x and y)
#  and returns a list of windows for the later vehicle detection
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    x_start_stop = get_x_start_stop(img, x_start_stop)
    y_start_stop = get_y_start_stop(img, y_start_stop)

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in reversed(range(ny_windows)):
        # print(" ys = ", ys)
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list


# Function taking an image and the list of windows to be searched (output of slide_windows())
#  and returns only the windows which encapsulate predicted cars
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    test_features_first = None

    print("\n XX0 -- len(windows) = ", len(windows))

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    first_pass = True
    second_pass = True
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = extract_features([test_img], read_img=False, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        filter_nan(features)
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        if first_pass:
            test_features_first = test_features
            first_pass = False
        elif second_pass:
            test_features_list = np.vstack((test_features_first, test_features))
            second_pass = False
        else:
            test_features_list = np.vstack((test_features_list, test_features))

    print("\n XX1 -- test_features_list.shape = ", test_features_list.shape)

    #6) Predict using our classifier
    # !! for optimization pass an array of features
    predictions = clf.predict(test_features_list)
    print("\n XX2 predictions = ", predictions)

    #7) If positive (prediction == 1) then save the window
    for index, prediction in enumerate(predictions):
        if prediction == 1:
            on_windows.append(windows[index])

    #8) Return windows for positive detections
    return on_windows


# Compute heatmap for a list of bounding boxes
def get_heatmap_hot_windows(bbox_list):
    heatmap = np.zeros((720,1280))
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


# set all pixels below a certain threshold to zero
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# compute the number of islands in a thresholded heatmap
#  i.e., how many different cars are found
def label_heat_map_islands(heatmap, threshold):
    heatmap = apply_threshold(heatmap, threshold)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    #plt.imshow(labels[0], cmap='gray')
    return labels


# return varius colors for various sized bounding boxes
def get_color_bb(size, divs):
    if size <= 48:
        return (125 / divs, 0, 0)
    if size <= 96:
        return (0, 180 / divs, 0)
    if size <= 144:
        return (0, 0, 150 / divs)
    if size <= 288:
        return (33 / divs, 34 / divs, 44 / divs)
    return (0, 0, 165 / divs)


# draw boxes around labeled pixel values
def draw_labeled_bboxes(img, labels, thick):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        x_size = bbox[1][0] - bbox[0][0]
        y_size = bbox[1][1] - bbox[0][1]

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], get_color_bb(x_size, 250.0), thick)
    # Return the image
    return img


# function to draw bounding boxes
def draw_boxes(img, bboxes, thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:

        x_size = bbox[1][0] - bbox[0][0]
        y_size = bbox[1][1] - bbox[0][1]
        # print(" bbox = ", bbox, " / x_size = ", x_size, " / y_size = ", y_size)
        # bbox =  ((100, 600), (196, 696))  ..  xy_window=(96, 96)

        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], get_color_bb(x_size, 250.0), thick)# get_color_bb(x_size), thick)
    # Return the image copy with boxes drawn
    return imcopy


# Helper functions to plot
def plot_cv(img):
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    plt.imshow(img2) # expect true color
    plt.show()


def my_plot(img):
    plt.imshow(img)
    plt.show()


def combo_plot_test_9_imgs(imgs):
    fig = plt.figure(figsize=(24, 9))

    ax1 = fig.add_subplot(331)
    ax1.imshow(imgs[0])

    ax2 = fig.add_subplot(332)
    ax2.imshow(imgs[1])

    ax3 = fig.add_subplot(333)
    ax3.imshow(imgs[2])

    ax4 = fig.add_subplot(334)
    ax4.imshow(imgs[3])

    ax5 = fig.add_subplot(335)
    ax5.imshow(imgs[4])

    ax6 = fig.add_subplot(336)
    ax6.imshow(imgs[5])

    ax7 = fig.add_subplot(337)
    ax7.imshow(imgs[6])

    ax8 = fig.add_subplot(338)
    ax8.imshow(imgs[7])

    ax9 = fig.add_subplot(339)
    ax9.imshow(imgs[8])

    fig.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def combo_plot_test_15_imgs(imgs):
    fig = plt.figure(figsize=(24, 9))

    ax1 = fig.add_subplot(3,5,1)
    ax1.imshow(imgs[0])

    ax2 = fig.add_subplot(3,5,2)
    ax2.imshow(imgs[1])

    ax3 = fig.add_subplot(3,5,3)
    ax3.imshow(imgs[2])

    ax4 = fig.add_subplot(3,5,4)
    ax4.imshow(imgs[3])

    ax5 = fig.add_subplot(3,5,5)
    ax5.imshow(imgs[4])

    ax6 = fig.add_subplot(3,5,6)
    ax6.imshow(imgs[5])

    ax7 = fig.add_subplot(3,5,7)
    ax7.imshow(imgs[6])

    ax8 = fig.add_subplot(3,5,8)
    ax8.imshow(imgs[7])

    ax9 = fig.add_subplot(3,5,9)
    ax9.imshow(imgs[8])

    ax10 = fig.add_subplot(3,5,10)
    ax10.imshow(imgs[9])

    ax11 = fig.add_subplot(3,5,11)
    ax11.imshow(imgs[10])

    ax12 = fig.add_subplot(3,5,12)
    ax12.imshow(imgs[11])

    ax13 = fig.add_subplot(3,5,13)
    ax13.imshow(imgs[12])

    ax14 = fig.add_subplot(3,5,14)
    ax14.imshow(imgs[14])

    ax15 = fig.add_subplot(3,5,15)
    ax15.imshow(imgs[14])

    fig.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def combo_plot_test_imgs(imgs):
    fig = plt.figure(figsize=(24, 9))

    ax1 = fig.add_subplot(231)
    ax1.imshow(imgs[0])

    ax2 = fig.add_subplot(232)
    ax2.imshow(imgs[1])

    ax3 = fig.add_subplot(233)
    ax3.imshow(imgs[2])

    ax4 = fig.add_subplot(234)
    ax4.imshow(imgs[3])

    ax5 = fig.add_subplot(235)
    ax5.imshow(imgs[4])

    ax6 = fig.add_subplot(236)
    ax6.imshow(imgs[5])

    fig.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def combo_plot(imgA, captionA, imgB, captionB):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(imgA)
    ax1.set_title(captionA, fontsize=50)

    ax2.imshow(imgB)
    ax2.set_title(captionB, fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

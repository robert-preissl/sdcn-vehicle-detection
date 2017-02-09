
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

# Read in our vehicles and non-vehicles
car1_img    = mpimg.imread("/Users/robert/KittyHawk/SDCN/vehicle_detection/vehicles/GTI_MiddleClose/image0000.png")
notcar1_img = mpimg.imread("/Users/robert/KittyHawk/SDCN/vehicle_detection/non-vehicles/GTI/image3861.png")

def combo_plot_test_imgs(gray, imgs):
    fig = plt.figure(figsize=(24, 9))

    ax1 = fig.add_subplot(221)
    ax1.imshow(gray, cmap='gray')
    ax1.set_title("Orig image", fontsize=50)

    ax2 = fig.add_subplot(222)
    ax2.imshow(imgs[0], cmap='gray')
    ax2.set_title("HOG CH0", fontsize=50)

    ax3 = fig.add_subplot(223)
    ax3.imshow(imgs[1], cmap='gray')
    ax3.set_title("HOG CH1", fontsize=50)

    ax4 = fig.add_subplot(224)
    ax4.imshow(imgs[2], cmap='gray')
    ax4.set_title("HOG CH2", fontsize=50)

    fig.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=False, feature_vector=feature_vec)
        return features


# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2

car1_img = notcar1_img

# Read in the image
gray_car1 = cv2.cvtColor(car1_img, cv2.COLOR_RGB2GRAY)
yuv_car1  = cv2.cvtColor(car1_img, cv2.COLOR_RGB2YUV)

hog_images = []
for channel in range(yuv_car1.shape[2]):
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(yuv_car1[:,:,channel], orient,
                            pix_per_cell, cell_per_block,
                            vis=True, feature_vec=False)

    hog_images.append(hog_image)

combo_plot_test_imgs(gray_car1, hog_images)
    # Plot the examples
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(gray_car1, cmap='gray')
    # plt.title('Example Car Image')
    # plt.subplot(122)
    # plt.imshow(hog_image, cmap='gray')
    # plt.title('HOG Visualization')
    # plt.show()


# done

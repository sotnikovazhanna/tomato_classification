import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2hsv
from skimage import exposure

IMG_SIZE = 224


def extract_color_histogram(image, bins=32):
    #Extract RGB and HSV color histograms
    hist_r = np.histogram(image[:, :, 0], bins=bins, range=(0, 255))[0]
    hist_g = np.histogram(image[:, :, 1], bins=bins, range=(0, 255))[0]
    hist_b = np.histogram(image[:, :, 2], bins=bins, range=(0, 255))[0]

    hsv = rgb2hsv(image / 255.0)
    hist_h = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 1))[0]
    hist_s = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 1))[0]
    hist_v = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 1))[0]

    return np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v])


def extract_texture_features(image):
    #Extract HOG texture features
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    hog_feat = hog(gray,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   visualize=False)

    return hog_feat


def extract_all_features(images):
    #Extract all features (color + texture) for a set of images
    features = []

    for i, img in enumerate(images):
        if i % 500 == 0:
            print(f"   Processed {i}/{len(images)} images")

        color_feat = extract_color_histogram(img)
        texture_feat = extract_texture_features(img)

        all_features = np.concatenate([color_feat, texture_feat])
        features.append(all_features)

    return np.array(features)
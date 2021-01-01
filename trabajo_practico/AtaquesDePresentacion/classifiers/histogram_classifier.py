import sklearn
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(img):
    hist = []
    img = cv.imread(img)
    color = ("b", "g", "r")
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        hist.append(histr)
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 256])
    # plt.show()
    hist = np.asarray(hist, dtype=np.float32)
    return np.ravel(hist)


def calculate_label(img):
    """
    1 is bonafide, otherwise 0
    """
    if "users" in img:
        return 1
    return 0


# def histogram_classifier():
#     return clf
import sys
from os import listdir

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from skimage.morphology import skeletonize


def rotateFilter(filter):
    return np.rot90(filter)


def thinnedImage(image, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))):
    img1 = image.copy()
    thin = np.zeros(image.shape, dtype="uint8")

    while cv2.countNonZero(img1) != 0:
        erode = cv2.erode(img1, kernel)
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        subset = erode - opening
        thin = cv2.bitwise_or(subset, thin)
        img1 = erode.copy()
    return thin


def erase(image):
    image = cv2.erode(image, np.ones((3, 3)))
    image = cv2.dilate(image, np.ones((2, 2)))
    image = cv2.dilate(image, np.ones((2, 2)))
    image = cv2.erode(image, np.ones((2, 2)))
    return image


def getTerminationBifurcation(image):
    newImage = image.copy()
    newImage = cv2.cvtColor(newImage, cv2.COLOR_GRAY2RGB)
    ret, image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i][j] == 1:
                cells = [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                    (1, 0),
                    (1, -1),
                    (0, -1),
                    (-1, -1),
                ]

                values = [image[i + l][j + k] for k, l in cells]
                values = np.reshape(values, 9).astype(np.int)
                crossings = 0
                for k in range(0, len(values) - 1):
                    crossings += abs(values[k] - values[k + 1])
                crossings //= 2
                if crossings == 2:
                    cv2.circle(
                        newImage,
                        (j - 1, i - 1),
                        radius=4,
                        color=(0, 0, 255),
                        thickness=1,
                    )
                if crossings > 3:
                    cv2.circle(
                        newImage,
                        (j - 1, i - 1),
                        radius=4,
                        color=(255, 0, 0),
                        thickness=1,
                    )
    return newImage


def skeletonization(img):
    # img = cv2.imread(iimg, 0)
    # cv2.imshow("original", img)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    # cv2.imshow("skel", skel)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img.copy()


def isBifurcation(row, col, img):
    summation = (
        int(img[row + 1][col - 1])
        + int(img[row + 1][col])
        + int(img[row + 1][col + 1])
        + int(img[row][col - 1])
        + int(img[row][col + 1])
        + int(img[row - 1][col - 1])
        + int(img[row - 1][col])
        + int(img[row - 1][col + 1])
    )
    if summation == 3:
        return True
    return False


def isEnding(row, col, img):
    summation = (
        int(img[row + 1][col - 1])
        + int(img[row + 1][col])
        + int(img[row + 1][col + 1])
        + int(img[row][col - 1])
        + int(img[row][col + 1])
        + int(img[row - 1][col - 1])
        + int(img[row - 1][col])
        + int(img[row - 1][col + 1])
    )
    if summation == 1:
        return True
    return False


if __name__ == "__main__":
    img = "./huellasFVC2004/101_3.tif"

    image = cv2.imread(img, 0)

    plt.matshow(image, cmap="gray")
    plt.title("original")
    fig1 = plt.figure(1)

    plt.matshow(np.invert(image), cmap="gray")
    plt.title("invert image")
    fig2 = plt.figure(2)

    image = cv2.adaptiveThreshold(
        np.invert(image), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    sklen = skeletonize(image)

    rows, cols = sklen.shape

    minutaes = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if sklen[i][j]:
                if isBifurcation(i, j, sklen):
                    minutaes.append([i, j, 3])

                if isEnding(i, j, sklen):
                    minutaes.append([i, j, 1])

    minucias = pd.DataFrame.from_records(minutaes)
    minucias.columns = ["row", "col", "label"]

    terminaciones = minucias[minucias["label"] == 1].drop(columns="label")
    bifurcaciones = minucias[minucias["label"] == 3].drop(columns="label")

    img = plt.imread(img)

    fig3 = plt.figure(3)

    plt.plot(terminaciones["col"], terminaciones["row"], "b.", label="terminaciones")
    plt.plot(bifurcaciones["col"], bifurcaciones["row"], "r.", label="bifurcaciones")

    plt.imshow(img, cmap="gray")
    plt.legend(loc="best")
    plt.title("minutaes")

    plt.show()

    # df.to_csv("minutaes.csv", sep=";")

    # plt.matshow(sklen, cmap="gray")
    # plt.title("skleton")
    # plt.show()

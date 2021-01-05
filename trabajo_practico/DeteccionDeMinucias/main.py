import sys
from os import listdir

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize


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

    # fig3 = plt.figure(3)
    # plt.matshow(sklen, cmap="gray")
    # plt.title("skleton")

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

    fig3 = plt.figure(4)

    plt.plot(terminaciones["col"], terminaciones["row"], "b.", label="terminaciones")
    plt.plot(bifurcaciones["col"], bifurcaciones["row"], "r.", label="bifurcaciones")

    plt.imshow(img, cmap="gray")
    plt.legend(loc="best")
    plt.title("minutaes")

    plt.show()

    # df.to_csv("minutaes.csv", sep=";")

    # plt.show()

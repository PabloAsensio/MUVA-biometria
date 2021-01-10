import os
import sys
from os import listdir

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import gabor
from skimage.morphology import skeletonize


# Eliminar parte de la imagen que se queda fuera de la huella
def getFingerprintRegion(img):
    row1 = 0
    row2 = 0
    col1 = 0
    col2 = 0
    for row in range(img.shape[0]):
        row1 = row
        if np.sum(img[row, 0 : img.shape[1]]) > 10:
            break

    for row in range(img.shape[0]):
        row2 = img.shape[0] - row - 1
        if np.sum(img[row2, 0 : img.shape[1]]) > 10:
            break

    for row in range(img.shape[1]):
        col1 = row
        if np.sum(img[0 : img.shape[0], col1]) > 10:
            break

    for row in range(img.shape[1]):
        col2 = img.shape[0] - row - 1
        if np.sum(img[0 : img.shape[0], col2]) > 10:
            break

    return row1, row2, col1, col2


def cutImage(img):
    kernel = np.ones((5, 5), dtype=np.uint8)

    _, imag_gabor = gabor(img, 0.7, theta=0)
    erosion = cv2.morphologyEx(imag_gabor, cv2.MORPH_OPEN, kernel)

    row1, row2, col1, col2 = getFingerprintRegion(erosion)

    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if not ((i < row2) and (i > row1) and (j > col1) and (j < col2)):
                if img[i, j] != 255:
                    img[i, j] = 255

    return img.copy(), row1, row2, col1, col2


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


def deginrad(degree):
    radiant = 2 * np.pi / 360 * degree
    return radiant


def gaborFilter(img):
    _, filt_imag = gabor(
        img,
        frequency=0.1,
        theta=deginrad(90),
        bandwidth=3,
        sigma_x=None,
        sigma_y=None,
        n_stds=3,
        offset=0,
        mode="reflect",
        cval=0,
    )
    return filt_imag.copy()


# Comprueba que no hay una minucia del mismo tipo en una vecindad 3x3
def isMinutaeClose(i, j, typeMinutae, l):
    if (
        [i - 1, j - 1, typeMinutae] in l
        or [i - 1, j, typeMinutae] in l
        or [i - 1, j + 1, typeMinutae] in l
        or [i + 1, j - 1, typeMinutae] in l
        or [i + 1, j, typeMinutae] in l
        or [i + 1, j + 1, typeMinutae] in l
        or [i, j - 1, typeMinutae] in l
        or [i, j + 1, typeMinutae] in l
    ):
        return True
    return False


def minutaeExtraction(img_path, plot=False):
    # se lee la imagen de la huella
    image = cv2.imread(img_path, 0)

    if plot:
        cv2.imshow("original", image)

    # se corta la imagen para que solo quede la huella, davuelve los margenes tambien
    image, row1, row2, col1, col2 = cutImage(image)
    if plot:
        cv2.imshow("cutted", image)

    # se aplica un filtlo de gabor que devuelve la imagen binarizada
    image = gaborFilter(image)
    img_gabor = image.copy()  # se guarda para mostras las minucias despues
    if plot:
        cv2.imshow("gabor", image)

    # se umbraliza la imagen adaptavivamente
    image = cv2.adaptiveThreshold(
        np.invert(image), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    if plot:
        cv2.imshow("gabor; adaptiveThreshold", image * 255)

    # se realiza un esqueleto de la huella
    sklen = skeletonize(image)
    if plot:
        plt.imshow(sklen, cmap="gray")
        plt.title("gabor; adaptiveThreshold; skeleton")

    # se cuentan las minucias dadas unas condiciones
    rows, cols = sklen.shape
    minutaes = []
    for i in range(rows):
        for j in range(cols):
            # se mira en aquellos puntos donde haya esqueleto
            if sklen[i, j]:
                # solo se miran los pixeles donde hay huella y formen parte del filtro de gabor (valen distinto de 0 ~False)
                if (
                    (i < row2)
                    and (i > row1)
                    and (j > col1)
                    and (j < col2)
                    and img_gabor[i][j]
                ):
                    if isBifurcation(i, j, sklen) and not isMinutaeClose(
                        i, j, 3, minutaes
                    ):
                        minutaes.append([i, j, 3])

                    if isEnding(i, j, sklen) and not isMinutaeClose(i, j, 1, minutaes):
                        minutaes.append([i, j, 1])

    minucias = pd.DataFrame.from_records(minutaes)
    minucias.columns = ["row", "col", "label"]

    terminaciones = minucias[minucias["label"] == 1].drop(columns="label")
    bifurcaciones = minucias[minucias["label"] == 3].drop(columns="label")

    if plot:
        img = plt.imread(img)

        _ = plt.figure(5)

        plt.plot(
            terminaciones["col"], terminaciones["row"], "b.", label="terminaciones"
        )
        plt.plot(
            bifurcaciones["col"], bifurcaciones["row"], "r.", label="bifurcaciones"
        )

        plt.imshow(img, cmap="gray")
        plt.legend(loc="best")
        plt.title("minutaes")

        plt.show()

    # se extrae el nombre del archivo
    img_name = img_path.split("/")[-1]
    img_name = "./Minutae_Extraction/" + img_name.replace(".tif", ".txt")

    # se guarda un txt con las minucias
    minucias.to_csv(img_name, sep=";")

    if plot:
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    plot = True
    imgs = [
        "./huellasFVC2004/101_3.tif",
        "./huellasFVC2004/101_8.tif",
        "./huellasFVC2004/106_3.tif",
        "./huellasFVC2004/110_7.tif",
    ]

    dirName = "Minutae_Extraction"
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    for img in imgs:
        minutaeExtraction(img)

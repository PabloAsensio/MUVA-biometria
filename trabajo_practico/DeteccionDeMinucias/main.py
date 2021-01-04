import cv2
import numpy as np
from os import listdir


def rotateFilter(filter):
    return np.rot90(filter)


def thinnedImage(image, kernel):
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
                if crossings == 1:
                    cv2.circle(
                        newImage,
                        (j - 1, i - 1),
                        radius=4,
                        color=(0, 0, 255),
                        thickness=1,
                    )
                if crossings == 3:
                    cv2.circle(
                        newImage,
                        (j - 1, i - 1),
                        radius=4,
                        color=(255, 0, 0),
                        thickness=1,
                    )
    return newImage


def skeletonization(iimg):
    img = cv2.imread(iimg, 0)
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

    cv2.imshow("skel", skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = "./huellasFVC2004/101_3.tif"
    skeletonization(img)

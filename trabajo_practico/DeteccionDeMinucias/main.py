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


def preprocessing(image):
    # cv2.imshow("ori", image)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("binarize", image)

    image = erase(image)
    # cv2.imshow("erase", image)

    """filter_VH = np.matrix('0, 1, 0;'
                          '0, 1, 0;'
                          '0, 1, 0').astype(np.uint8)


    filter_diag = np.matrix('1, 0, 0;'
                            '0, 1, 0;'
                            '0, 0, 1').astype(np.uint8)

    filter_VH = np.asarray(filter_VH)
    filter_diag = np.asarray(filter_diag)

    #filter_VH = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
    #filter_diag = cv2.imread('template2.jpg', cv2.IMREAD_GRAYSCALE)

    image_v = cv2.erode(image,filter_VH)
    filter_VH = rotateFilter(filter_VH)
    image_h = cv2.erode(image,filter_VH)

    image_diag = cv2.erode(image,filter_diag)
    filter_diag = rotateFilter(filter_diag)
    image_diag2 = cv2.erode(image,filter_diag)

    match = image_v + image_h + image_diag + image_diag2

    cv2.imshow('mathc', match)"""

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    testImage = thinnedImage(image, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    testImage = testImage + thinnedImage(testImage, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = testImage + thinnedImage(image, kernel)
    # cv2.imshow("thinned", image)

    # image = getTerminationBifurcation(image)
    # cv2.imshow("terminations", image)

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.imshow('rgb', image)

    # bifur = np.matrix('1, 0, 0, 0;'
    #                 '0, 1, 1, 1;'
    #                '1, 0, 0, 0').astype(np.uint8)
    """newImage = image.copy()
    bifur = cv2.imread('bifurcacion.jpg')
    ret ,bifur = cv2.threshold(bifur, 150, 255, cv2.THRESH_BINARY )
    h, w, canals = bifur.shape
    bifur45 = cv2.imread('bifurcacion45.jpg')
    ret ,bifur45 = cv2.threshold(bifur45, 150, 255, cv2.THRESH_BINARY )
    h45, w45, canals = bifur45.shape
    for angle in range(0,2):

        bifur = rotateFilter(bifur)
        bifur45 = rotateFilter(bifur45)
        cv2.imshow('bifur', bifur)
        res = cv2.matchTemplate(image, bifur, cv2.TM_CCORR_NORMED)
        threshhold = 0.6
        loc = np.where(res >= threshhold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(newImage, pt, (pt[0] + h, pt[1] + w), (255, 0, 0), 2)
        res = cv2.matchTemplate(image, bifur45, cv2.TM_CCORR_NORMED)
        threshhold = 0.6
        loc = np.where(res >= threshhold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(newImage, pt, (pt[0] + h, pt[1] + w), (255, 0, 0), 2)

    terminaciones = cv2.imread('terminaciones.jpg')
    ret ,terminaciones = cv2.threshold(terminaciones, 150, 255, cv2.THRESH_BINARY )
    h, w, canals = bifur.shape
    terminaciones45 = cv2.imread('terminaciones45.jpg')
    ret ,terminaciones45 = cv2.threshold(terminaciones45, 150, 255, cv2.THRESH_BINARY )
    h45, w45, canals = terminaciones45.shape
    for angle in range(0,2):

        terminaciones = rotateFilter(terminaciones)
        terminaciones45 = rotateFilter(terminaciones45)
        cv2.imshow('bifur', terminaciones)
        res = cv2.matchTemplate(image, terminaciones, cv2.TM_CCORR_NORMED)
        threshhold = 0.5
        loc = np.where(res >= threshhold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(newImage, pt, (pt[0] + h, pt[1] + w), (0, 0, 255), 2)
        res = cv2.matchTemplate(image, terminaciones45, cv2.TM_CCORR_NORMED)
        threshhold = 0.4
        loc = np.where(res >= threshhold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(newImage, pt, (pt[0] + h, pt[1] + w), (0, 0, 255), 1)



    cv2.imshow('rectangles', newImage)"""
    cv2.imshow("processed", image)
    cv2.waitKey()


images = []
path = "./huellasFVC2004/"
for file in listdir(path):
    images.append(cv2.imread(path + file, cv2.IMREAD_GRAYSCALE))

images = np.asarray(images)
for image in images:
    preprocessing(image)


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

    cv2.imshow("skel", skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = "./huellasFVC2004/101_3.tif"
    preprocessing(img)
    # skeletonization(img)

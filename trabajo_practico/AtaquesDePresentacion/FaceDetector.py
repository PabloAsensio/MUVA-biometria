import numpy as np
import cv2

from os import listdir
from os.path import isfile, isdir
import os
import re

def getSegmentedFaces(path):
    faces = []
    tags = np.array([])
    for file in listdir(path):

        if isfile(path + '/' + file):
            faces_seg, tags_seg = caras(path +'/', file)
            if len(faces_seg) != 0:
                faces = faces + faces_seg
                tags = np.append(tags, tags_seg)

        elif isdir(path + '/' + file):
            faces_rec, tags_rec = getSegmentedFaces(path+'/'+ file)
            faces = faces + faces_rec
            tags = np.append(tags, tags_rec)
    return faces, tags

def caras(path, file):
    face_cascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    img = cv2.imread(path+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_img = []
    tags = np.array([])
    for (x,y,w,h) in faces:
        image = img[y:y+h, x:x+w]
        faces_img.append(cv2.resize(image, (300, 300)))
        if re.findall('attack*', os.path.basename(path[:-1])):
            tags = np.append(tags, 1)
        else:
            tags = np.append(tags, 0)
    if len(faces_img) != len(tags):
        print()
    return faces_img, tags





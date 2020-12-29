import numpy as np
import cv2

from os import listdir
from os.path import isfile, join, isdir
import os
import re

string_change = 'D:/Descargas/REAL_SENSE_segmented/REAL_SENSE/REAL_SENSE_IR/'
def findRecursive(path):
    for file in listdir(path):
        if isfile(path + '/' + file):
            caras(path +'/', file)
        elif isdir(path + '/' + file):
            newPath = path.replace('D:/Descargas/REAL_SENSE/REAL_SENSE/REAL_SENSE_IR/', string_change)
            if not os.path.exists(newPath):
                os.mkdir(newPath)
            findRecursive(path+'/'+ file)

def caras(path, file):
    face_cascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')
    img = cv2.imread(path+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = 0
    images = np.empty(40)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        newImage = img[y:y+h, x:x+w]
        newImage = cv2.resize(newImage, (1000, 1000))
        newPath = path.replace('D:/Descargas/REAL_SENSE/REAL_SENSE/REAL_SENSE_IR/', string_change)
        if not os.path.exists(newPath):
            os.mkdir(newPath)
        cv2.imwrite(newPath+'/'+str(count) + file, img[y:y+h, x:x+w])
        count += 1


findRecursive('D:/Descargas/REAL_SENSE/REAL_SENSE/REAL_SENSE_IR/')


'''face_cascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')

path = 'D:/Descargas/REAL_SENSE/REAL_SENSE/REAL_SENSE_IR/USER_024/attack_01/'
for file in listdir(path):
    img = cv2.imread(path+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    newImage = cv2.resize(img,(1000,1000))
    #cv2.imshow('im', newImage)
    #cv2.waitKey()
    count = 0
    images = np.empty(40)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        newImage = img[y:y+h, x:x+w]
        newImage = cv2.resize(newImage, (1000, 1000))
        if not os.path.exists('D:/Descargas/FravAttack_con_bonafide/FravAttack/segmented/'+os.path.basename(path)):
            os.mkdir('D:/Descargas/FravAttack_con_bonafide/FravAttack/segmented/'+os.path.basename(path)+'/')
        cv2.imwrite('D:/Descargas/FravAttack_con_bonafide/FravAttack/segmented/'+os.path.basename(path)+'/'+str(count) + file, img[y:y+h, x:x+w])
        count += 1
        #cv2.imshow('segmented', newImage)
        #cv2.waitKey()
        #Pintado del rectangulo en la cara'''
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]'''
        #Deteccion de ojos
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
#            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)'''


    #img = cv2.resize(img, (1000,1000))
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()'''



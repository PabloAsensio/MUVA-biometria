import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')


img = cv2.imread("D:/Descargas/FravAttack_con_bonafide/FravAttack/SONY/RAW/FRONT/USER/USUARIO_000.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
newImage = cv2.resize(img,(1000,1000))
cv2.imshow('im', newImage)
cv2.waitKey()

images = np.empty(40)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    newImage = img[y:y+h, x:x+w]
    newImage = cv2.resize(newImage, (1000, 1000))
    cv2.imshow('segmented', newImage)
    cv2.waitKey()
    #Pintado del rectangulo en la cara
    '''img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]'''
    #Deteccion de ojos
    '''eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)'''


img = cv2.resize(img, (1000,1000))
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2 as cv
img = cv.imread('lena.jpg',0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('clahe_equalization.jpg',cl1)
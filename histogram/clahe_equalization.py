import numpy as np
import cv2 as cv

img = cv.imread('image_to_improve.png')
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

lab_clahe = cv.merge((l_clahe, a, b))
bgr_clahe = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)

cv.imwrite('clahe_equalization_color.jpg', bgr_clahe)
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg')

# Box filter averages a kxk window; fast, takes kernel size tuple, but smears edges heavily.
#blur = cv2.blur(img,(5,5))  
# Gaussian kernel weighted by distance; needs kernel and sigma, good generic smooth but still softens edges.
#blur = cv2.GaussianBlur(img,(5,5),0)
# Replaces pixel with median in window size k; single odd int, great for salt-and-pepper noise, slower on large kernels.
#blur = cv2.medianBlur(img,5)
# Combines spatial/color sigmas; expects diameter, sigmaColor, sigmaSpace, preserves edges yet computationally expensive.
blur = cv2.bilateralFilter(img,9,75,75)  

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

plt.subplot(121); plt.imshow(img_rgb); plt.title('Original'); plt.axis('off')
plt.subplot(122); plt.imshow(blur_rgb); plt.title('Blurred'); plt.axis('off')
plt.show()

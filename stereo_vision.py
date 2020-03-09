import numpy as np
import cv2
from matplotlib import pyplot as plt


image = cv2.imread('left_image.jpg',0)
image2 = cv2.imread('right_image.jpg',0)

final_wide = 350
r = float(final_wide) / image.shape[1]
dim = (final_wide, int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA))
resized2 = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(resized,resized2)
plt.imshow(disparity,'gray')
plt.show()

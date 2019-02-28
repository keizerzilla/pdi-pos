import cv2
import numpy as np
from matplotlib import pyplot as plt

images = ["cameraman.jpg", "crowd2.jpg", "lenna.jpg", "tutu.jpg"]

for i in images:
	img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
	mod = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
	
	ret, thresh1 = cv2.threshold(mod, 127, 255, cv2.THRESH_BINARY)
	ret, thresh2 = cv2.threshold(mod, 127, 255, cv2.THRESH_BINARY_INV)
	ret, thresh3 = cv2.threshold(mod, 127, 255, cv2.THRESH_TRUNC)
	ret, thresh4 = cv2.threshold(mod, 127, 255, cv2.THRESH_TOZERO)
	ret, thresh5 = cv2.threshold(mod, 127, 255, cv2.THRESH_TOZERO_INV)
	
	titles = ['Raw','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
	images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
	
	for i in range(6):
		plt.subplot(2, 3, i+1)
		plt.imshow(images[i], "gray")
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])
	
	plt.show()


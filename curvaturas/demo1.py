import warnings
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

warnings.filterwarnings("ignore")

def convolve(image, kernel):
	isMedian = False
	isMode = False
	
	if isinstance(kernel, str):
		isMedian = kernel == "median"
		isMode = kernel == "mode"
		kernel = np.ones((3, 3), dtype="int")
	
	(ih, iw) = image.shape[:2]
	(kh, kw) = kernel.shape[:2]
	pad = (kw - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((ih, iw), dtype="float32")
	
	for y in np.arange(pad, ih + pad):
		for x in np.arange(pad, iw + pad):
			roi = image[y-pad:y+pad+1, x-pad:x+pad+1]
			
			k = 0			
			if isMedian:
				k = np.median(roi)
			elif isMode:
				k = mode(roi, axis=None)[0]
			else:
				k = (roi * kernel).sum()
			
			output[y-pad, x-pad] = k
	
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	
	return output

def k_curvature(im, k=3):
	zeros = np.pad(im, k, 'constant', constant_values=0)
	kurv_img = np.zeros_like(im)
	
	rows, cols = im.shape
	for i in range(rows):
		for j in range(cols):
			alpha = zeros[i + k, j] - zeros[i - k, j]
			epslon = zeros[i, j + k] - 2*zeros[i, j] + zeros[i, j - k]
			gamma = zeros[i, j + k] - zeros[i, j - k]
			delta = zeros[i + k, j] - 2*zeros[i, j] + zeros[i - k, j]
			
			kv = (alpha*epslon - gamma*delta)/(alpha*alpha + gamma*gamma)**(3/2)
			if math.isnan(kv):
				kv = 0
			
			kurv_img[i, j] = kv
	
	return kurv_img

def kurv_plot(corners):
	x = np.ravel(corners)
	normalized = (x-min(x))/(max(x)-min(x))
	
	return normalized

if __name__ == "__main__":
	im = cv2.imread("../rect.bmp")
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	laplacian = np.array(([0,  1, 0], [1, -4, 1], [0,  1, 0]), dtype="int")
	
	ret, tresh = cv2.threshold(imgray, 127, 255, 0)
	
	tresh = convolve(tresh, laplacian)
	tresh = cv2.GaussianBlur(tresh, (3, 3), 0)
	
	contours, _ = cv2.findContours(tresh.copy(),
		                           cv2.RETR_EXTERNAL,
		                           cv2.CHAIN_APPROX_NONE)
	s = np.zeros_like(tresh)
	cv2.drawContours(s, contours, -1, (255, 255, 255), 1)
	
	plt.subplot(1, 3, 1)
	plt.imshow(imgray, cmap="gray")
	plt.title("original")
	plt.subplot(1, 3, 2)
	plt.imshow(tresh, cmap="gray")
	plt.title("limiarização")
	plt.subplot(1, 3, 3)
	plt.imshow(s, cmap="gray")
	plt.title("contornos encontrados")
	plt.show()
	plt.close()
	
	count = 1
	for k in [1, 4, 8, 12]:
		corners = k_curvature(s, k)
		
		plt.subplot(2, 4, count)
		plt.imshow(corners, cmap="gray")
		plt.title("cantos (k = {})".format(k))
		
		plt.subplot(2, 4, count+4)
		normalized = kurv_plot(corners)
		plt.plot(normalized)
		plt.title("curvatura (k = {})".format(k))
		
		print("k = {} ok!".format(k))
		count += 1
	
	plt.show()
	

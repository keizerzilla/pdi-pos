import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def k_curvature(im, k=3):
	zeros = np.pad(im, k, 'constant', constant_values=0)
	#zeros = zeros.astype(float)
	kurv_img = np.zeros_like(im)
	
	rows, cols = im.shape
	for i in range(rows):
		for j in range(cols):
			alpha = zeros[i + k, j] - zeros[i - k, j]
			epslon = zeros[i, j + k] - 2*zeros[i, j] + zeros[i, j - k]
			gama = zeros[i, j + k] - zeros[i, j - k]
			delta = zeros[i + k, j] - 2*zeros[i, j] + zeros[i - k, j]
			
			kurv = (alpha*epslon - gama*delta) / (alpha*alpha + gama*gama)**(3/2)
			if math.isnan(kurv):
				kurv = 0
			
			kurv_img[i, j] = kurv
	
	return kurv_img
	

def plot_figs(fig1, fig2, suptitle, path=None, t1=None, t2=None):
	plt.subplot(1, 2, 1)
	plt.imshow(fig1, cmap="gray")
	if t1: plt.title(t1)
	plt.subplot(1, 2, 2)
	plt.imshow(fig2, cmap="gray")
	if t2: plt.title(t2)
	
	plt.suptitle(suptitle)
	
	if path != None:
		plt.savefig(path)
	else:
		plt.show()
	
	plt.close()

if __name__ == "__main__":
	im = cv2.imread("../rect1.bmp")
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	ret, tresh = cv2.threshold(imgray, 127, 255, 0)
	
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
	
	corners = k_curvature(s)
	
	plt.imshow(corners, cmap="gray")
	plt.show()
	plt.close()
	
	x = np.ravel(corners)
	normalized = (x-min(x))/(max(x)-min(x))
	
	plt.plot(normalized)
	plt.show()
	

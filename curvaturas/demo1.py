import cv2
import numpy as np
import matplotlib.pyplot as plt

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
	#im = cv2.imread("../lenna.jpg")
	im = cv2.imread("../rect1.bmp")
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, tresh = cv2.threshold(imgray, 127, 255, 0)
	contours, _ = cv2.findContours(tresh.copy(),
		                           cv2.RETR_EXTERNAL,
		                           cv2.CHAIN_APPROX_SIMPLE)
	im2 = im.copy()
	cv2.drawContours(im2, contours, -1, (0, 255, 0), 4)
	
	print(contours)
	ans = tresh[tuple(contours)]
	print(ans)
	
	"""
	plt.subplot(1, 3, 1)
	plt.imshow(imgray, cmap="gray")
	plt.title("original")
	plt.subplot(1, 3, 2)
	plt.imshow(tresh, cmap="gray")
	plt.title("limiarização")
	plt.subplot(1, 3, 3)
	plt.imshow(im2, cmap="gray")
	plt.title("contornos encontrados")
	plt.show()
	plt.close()
	"""

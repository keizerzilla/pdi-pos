import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../crowd2.jpg", cv2.IMREAD_GRAYSCALE)
cA = img
fig = plt.figure()
for i in range(4):
	cA, (cH, cV, cD) = pywt.dwt2(cA, "haar")
	
	# cA
	ax = fig.add_subplot(4, 4, i + 1)
	ax.imshow(cA, interpolation="nearest", cmap=plt.cm.gray)
	ax.set_title("Aproximação {}".format(i + 1), fontsize=10)
	ax.set_xticks([])
	ax.set_yticks([])
	
	# cH
	ax = fig.add_subplot(4, 4, i + 5)
	ax.imshow(cH, interpolation="nearest", cmap=plt.cm.gray)
	ax.set_title("Horizontal {}".format(i + 1), fontsize=10)
	ax.set_xticks([])
	ax.set_yticks([])
	
	# cV
	ax = fig.add_subplot(4, 4, i + 9)
	ax.imshow(cV, interpolation="nearest", cmap=plt.cm.gray)
	ax.set_title("Vertical {}".format(i + 1), fontsize=10)
	ax.set_xticks([])
	ax.set_yticks([])
	
	# cD
	ax = fig.add_subplot(4, 4, i + 13)
	ax.imshow(cD, interpolation="nearest", cmap=plt.cm.gray)
	ax.set_title("Diagonal {}".format(i + 1), fontsize=10)
	ax.set_xticks([])
	ax.set_yticks([])

fig.tight_layout()
plt.show()


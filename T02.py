import cv2
import numpy as np
from scipy.stats import mode
from skimage.exposure import rescale_intensity

# Cria copia de uma imagem com adicao de ruido gaussiano
def add_gaussian_noise(image, mean=0.0, var=1.0):
	row, col = image.shape
	sigma = var**0.5
	gauss = np.random.normal(mean, sigma, (row, col))
	gauss = gauss.reshape(row, col)
	noisy = image + gauss
	
	return noisy

# Aplica convolucao de um kernel em uma imagem alvo
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

images = ["cameraman.jpg", "crowd2.jpg", "lenna.jpg", "tutu.jpg"]

laplacian = np.array((
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]), dtype="int")

gaussian = np.array((
           [1.0/16.0, 2.0/16.0, 1.0/16.0],
           [2.0/16.0, 4.0/16.0, 2.0/16.0],
           [1.0/16.0, 2.0/16.0, 1.0/16.0]), dtype="float")

kernels = {
	"laplacian" : laplacian,
	"gaussian" : gaussian,
	"median" : "median",
	"mode" : "mode"
}

for i in images:
	for name, kernel in kernels.items():
		img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
		noisy = add_gaussian_noise(img, 0.0, 10.0)

		print("aplicando kernel {} em {}".format(name, i))
		ans_raw = convolve(img, kernel)
		ans_noisy = convolve(noisy, kernel)
		
		impath_raw = "T02/" + i.replace(".jpg", "_{}.jpg".format(name))
		impath_noisy = "T02/" + i.replace(".jpg", "_noisy.jpg")
		impath_cnoisy = "T02/" + i.replace(".jpg", "_noisy_{}.jpg".format(name))
		cv2.imwrite(impath_raw, ans_raw)
		cv2.imwrite(impath_noisy, noisy)
		cv2.imwrite(impath_cnoisy, ans_noisy)
		

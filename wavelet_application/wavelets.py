import cv2
import pywt
import numpy as np
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

img = cv2.imread("../crowd2.jpg", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (3, 3), 0)

for w in pywt.wavelist(kind="discrete"):
	wavelet = pywt.Wavelet(w)
	
	coeffs = pywt.wavedec2(blur, wavelet)
	#cA, (cH, cV, cD) = coeffs
	
	rec = pywt.waverec2(coeffs, wavelet)
	
	mse = compare_mse(blur, rec)
	psnr = compare_psnr(blur, rec)
	ssim = compare_ssim(blur, rec)
	
	print(w, ">>", mse, psnr, ssim)
	

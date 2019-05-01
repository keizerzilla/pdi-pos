import cv2
import numpy as np
import matplotlib.pyplot as plt

def euler(omega):
	return np.exp(1j*omega)

def real2complex(mag, ang):
	return mag * np.exp(1j*ang)

def imshow(img):
	plt.imshow(img, cmap="gray")
	plt.show()
	plt.close()

img = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
shift = int(img.shape[0]/2)
shifted = np.roll(img, shift)
shifted[:, 0:shift] = 0

fft2 = np.fft.fft2(img)
angle = np.angle(fft2)
magnitude = np.abs(fft2)

omega = np.pi
new_angle = angle + omega

lol = real2complex(magnitude, new_angle)
imshow(np.fft.ifft2(lol).real)

t = img * euler(-1*omega)
imshow(t.real)

"""
#factor = euler(3.14*shift, shift)
factor = real2complex(magnitude, angle*shift)
shifted_fft2 = fft2 * factor

print(shifted_fft2.dtype)

inv_shifted = np.fft.ifft2(shifted_fft2)
imshow(inv_shifted.real)
"""


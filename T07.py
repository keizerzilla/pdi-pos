import cv2
import numpy as np
import matplotlib.pyplot as plt

def real2complex(mag, ang):
	return mag * np.exp(1j*ang)

def show_figs(fig1, fig2, suptitle, t1, t2):
	plt.subplot(1, 2, 1)
	plt.imshow(fig1, cmap="gray")
	plt.title(t1)
	
	plt.subplot(1, 2, 2)
	plt.imshow(fig2, cmap="gray")
	plt.title(t2)
	
	plt.suptitle(suptitle)
	plt.show()
	plt.close()

tutu = cv2.imread("tutu.jpg", cv2.IMREAD_GRAYSCALE)
lenna = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)

show_figs(tutu,
	      lenna,
	      "Imagens de teste",
	      "tutu.jpg",
	      "lenna.jpg")

tutu_fft = np.fft.fft2(tutu)
lenna_fft = np.fft.fft2(lenna)

tutu_abs = np.abs(tutu_fft)
lenna_abs = np.abs(lenna_fft)

tutu_shift = np.log(np.fft.fftshift(tutu_abs))
lenna_shift = np.log(np.fft.fftshift(lenna_abs))

show_figs(tutu_shift,
	      lenna_shift,
	      "Log das magnitudes das transformadas",
	      "magnitude tutu",
	      "magnitude lenna")
	
tutu_angle = np.angle(tutu_fft)
lenna_angle = np.angle(lenna_fft)

show_figs(tutu_angle,
	      lenna_angle,
	      "Fase das transformadas",
	      "fase tutu",
	      "fase lenna")

c1 = real2complex(tutu_abs, lenna_angle)
c2 = real2complex(lenna_abs, tutu_angle)

ans1 = np.fft.ifft2(c1).real
ans2 = np.fft.ifft2(c2).real

show_figs(ans1,
	      ans2,
	      "Inversa das trocas de magnitude e fase",
	      "T07/04-inversa-troca-mag-fase.jpg",
	      "magnitude tutu + fase lenna",
	      "magnitude lenna + fase tutu")


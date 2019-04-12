import cv2
import numpy as np
import matplotlib.pyplot as plt

def polar2rad(mag, ang):
	return mag * np.exp(1j*ang)

def conv_uint8(data):
	return data.astype(np.uint8)

def log_view(data):
	return 20*np.log(data)

def save_figs(fig1, fig2, suptitle, path, t1=None, t2=None):
	plt.subplot(1, 2, 1)
	plt.imshow(fig1, cmap="gray")
	if t1: plt.title(t1)
	plt.subplot(1, 2, 2)
	plt.imshow(fig2, cmap="gray")
	if t2: plt.title(t2)
	
	plt.suptitle(suptitle)
	plt.savefig(path)
	plt.close()

if __name__ == "__main__":
	tutu = cv2.imread("tutu.jpg", cv2.IMREAD_GRAYSCALE)
	lenna = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)

	save_figs(tutu,
		      lenna,
		      "Imagens de teste",
		      "T07/01-imagens-de-teste.jpg",
		      "tutu.jpg",
		      "lenna.jpg")

	tutu_fft = np.fft.fft2(tutu)
	lenna_fft = np.fft.fft2(lenna)

	tutu_abs = np.abs(tutu_fft)
	lenna_abs = np.abs(lenna_fft)

	tutu_shift = log_view(np.fft.fftshift(tutu_abs))
	lenna_shift = log_view(np.fft.fftshift(lenna_abs))

	save_figs(tutu_shift,
		      lenna_shift,
		      "Log das magnitudes das transformadas",
		      "T07/02-magnitudes-fft2.jpg",
		      "magnitude tutu",
		      "magnitude lenna")
	
	tutu_angle = np.angle(tutu_fft)
	lenna_angle = np.angle(lenna_fft)
	
	save_figs(tutu_angle,
		      lenna_angle,
		      "Fase das transformadas",
		      "T07/03-fase-fft2.jpg",
		      "fase tutu",
		      "fase lenna")
	
	t1 = polar2rad(tutu_abs, lenna_angle)
	t2 = polar2rad(lenna_abs, tutu_angle)
	
	ans1 = np.fft.ifft2(t1).real
	ans2 = np.fft.ifft2(t2).real
	
	save_figs(ans1,
		      ans2,
		      "Inversa das trocas de magnitude e fase",
		      "T07/04-inversa-troca-mag-fase.jpg",
		      "magnitude tutu + fase lenna",
		      "magnitude lenna + fase tutu")


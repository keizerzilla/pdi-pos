import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_figs(fig1, fig2, suptitle, path, t1=None, t2=None):
	plt.subplot(1, 2, 1)
	plt.imshow(fig1, cmap="gray")
	if t1: plt.title(t1)
	plt.subplot(1, 2, 2)
	plt.imshow(fig2, cmap="gray")
	if t2: plt.title(t2)s
	
	plt.suptitle(suptitle)
	plt.savefig(path)
	plt.close()

if __name__ == "__main__":
	lenna = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
	lenna_fft = np.fft.fft2(lenna)
	lenna_conj = np.conj(lenna_fft)
	lenna_inv = np.fft.ifft2(lenna_conj).real

	save_figs(lenna,
		      lenna_inv,
		      "Efeito do complexo conjulgado da TF em uma imagem",
		      "T06/01-efeito-conjultado-fft.jpg",
		      "imagem original",
		      "inversa do conjulgado")


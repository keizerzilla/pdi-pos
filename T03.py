import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin

def reconstruct(centers, labels, shape):
	ans = np.zeros(shape, dtype=np.uint8)
	label_idx = 0
	for i in range(shape[0]):
		for j in range(shape[1]):
			ans[i][j] = centers[labels[label_idx]]
			label_idx += 1
	
	return ans

def cluster_kmeans(data, k=8, metric="euclidean"):
	img = np.array(data, dtype=np.float64) / 255
	img = np.reshape(data, (data.shape[0]*data.shape[1], 3))
	
	centers = shuffle(img, random_state=0)[:k]
	
	while True:
		labels = pairwise_distances_argmin(centers, img, metric=metric, axis=0)
		new_centers = np.array([img[labels == i].mean(0) for i in range(k)])
		
		if np.all(centers == new_centers):
			break
		
		centers = new_centers
	
	return reconstruct(centers, labels, data.shape)

if __name__ == "__main__":
	image = cv2.imread("lena_std.tif")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	n_clusters = [4, 8, 16, 32]
	metrics = ["euclidean", "manhattan", "cosine", "chebyshev"]
	fig, axes = plt.subplots(4, 4)
	row = 0
	for k in n_clusters:
		col = 0
		for metric in metrics:
			ans = cluster_kmeans(image, k=k, metric=metric)
			axes[row, col].imshow(ans)
			axes[row, col].set_title("{}, k={}".format(metric, k))
			col += 1
			print(metric, k, "ok")
		row += 1

	plt.show()

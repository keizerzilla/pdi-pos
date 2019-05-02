import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("dataset_funceme_raspado.csv")
df = df.drop(["ano", "mes", "dia31"], axis=1)

pca = PCA()
pca.fit(df)
tdata = pca.transform(df)

i = 1
for c, v in zip(pca.components_, pca.singular_values_):
	plt.subplot(5, 6, i)
	plt.plot(c)
	plt.title("{}".format(round(v, 4)))
	i += 1

plt.show()


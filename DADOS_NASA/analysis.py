import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("dataset_funceme_raspado.csv")

data = []
for d in range(12):
	mes = d+1
	l = [mes] + list(df[df["mes"] == mes].mean().drop(["ano", "mes"]))
	data.append(l)

header = ["mes"] + [str(d+1) for d in range(31)]
tf = pd.DataFrame(data, columns=header)
"""
i=1
for index, row in tf.iterrows():
	d = row.drop(["mes"])
	
	plt.subplot(3, 4, i)
	d.plot()
	
	i += 1

plt.show()
plt.close()
"""
tf = tf.drop(["mes"], axis=1)
pca = PCA()
pca.fit(tf)

i=1
for c in pca.components_:
	plt.subplot(3, 4, i)
	plt.plot(c)
	i += 1

plt.show()
plt.close()


import pandas as pd

data = []
with open("dataset_funceme.txt") as dataset:
	lines = dataset.readlines()
	for l in lines:
		newline = l.rstrip('\n')
		dataline = newline.split()
		n = 33 - len(dataline)
		dataline = dataline + ["0,0" for i in range(n)]
		
		dataline[0] = int(dataline[0])
		dataline[1] = int(dataline[1])
		for i in range(31):
			dataline[i+2] = float(dataline[i+2].replace(",", "."))
		
		data.append(dataline)

header = ["ano", "mes"] + ["dia{}".format(d+1) for d in range(31)]
df = pd.DataFrame(data, columns=header)
df.to_csv("dataset_funceme_raspado.csv", index=None)

# dataset com meses como atributos
data = df.drop(["ano"], axis=1)
transp = {d+1 : [] for d in range(12)}

for index, row in data.iterrows():
	m = row["mes"]
	s = list(row.drop(["mes"]))
	transp[m] += s

transp = pd.DataFrame(transp)
transp.to_csv("dataset_funceme_pormes.csv", index=None)


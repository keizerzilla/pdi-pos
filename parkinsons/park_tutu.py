import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC as SVM
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid as NC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RandomForest

def mode(predictions):
	return max(set(predictions), key=predictions.count)

warnings.filterwarnings("ignore")

classifiers = {
	"Naive Bayes"           : NaiveBayes(),
	"Logistic Regression"   : LogisticRegression(),
	"k-NN"                  : KNN(p=1, n_neighbors=1),
	"Multilayer Perceptron" : MLP(),
	"Random Forest"         : RandomForest(n_estimators=100),
	"SVM (Linear)"          : SVM(kernel="linear", gamma="auto"),
	"SVM (RBF)"             : SVM(kernel="rbf", gamma="auto")
}

scores = {
	"subject"               : [],
	"Naive Bayes"           : [],
	"Logistic Regression"   : [],
	"k-NN"                  : [],
	"Multilayer Perceptron" : [],
	"Random Forest"         : [],
	"SVM (Linear)"          : [],
	"SVM (RBF)"             : []
}

f1s = {
	"subject"               : [],
	"Naive Bayes"           : [],
	"Logistic Regression"   : [],
	"k-NN"                  : [],
	"Multilayer Perceptron" : [],
	"Random Forest"         : [],
	"SVM (Linear)"          : [],
	"SVM (RBF)"             : []
}

mccs = {
	"subject"               : [],
	"Naive Bayes"           : [],
	"Logistic Regression"   : [],
	"k-NN"                  : [],
	"Multilayer Perceptron" : [],
	"Random Forest"         : [],
	"SVM (Linear)"          : [],
	"SVM (RBF)"             : []
}

voting = {
	"subject" : [],
	"voted"   : [],
	"true"    : []
}

df = pd.read_csv("parkinsons.csv")
df = df.drop(["gender"], axis=1)

for i in range(252):
	print("SUBJECT {}".format(i))
	
	scores["subject"].append(i)
	f1s["subject"].append(i)
	mccs["subject"].append(i)
	
	train_set = df.loc[df["id"] != i].drop(["id"], axis=1)
	test_set = df.loc[df["id"] == i].drop(["id"], axis=1)
	
	X_train = train_set.drop(["class"], axis=1)
	y_train = train_set["class"]
	X_test = test_set.drop(["class"], axis=1)
	y_test = test_set["class"]
	
	scaler = StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	
	pca = PCA()
	pca.fit(X_train)
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	
	predictions = []
	for name, classifier in classifiers.items():
		classifier.fit(X_train_pca, y_train)
		pred = classifier.predict(X_test_pca)
		
		score = round(accuracy_score(y_test, pred), 2)
		f1 = round(f1_score(y_test, pred), 2)
		mcc = round(matthews_corrcoef(y_test, pred), 2)
		
		scores[name].append(score)
		f1s[name].append(f1)
		mccs[name].append(mcc)
		
		predictions.extend(list(pred))
		
		print("{:<25}{} {} {}".format(name, score, f1, mcc))
	
	voted_label = mode(predictions)
	true_label = list(y_test)[0]
	
	voting["subject"].append(i)
	voting["voted"].append(voted_label)
	voting["true"].append(true_label)
	
	print("Voted/True: {}/{}".format(voted_label, true_label))
	print()
	
scores = pd.DataFrame(scores)
scores.to_csv("scores.csv", index=None)

f1s = pd.DataFrame(f1s)
f1s.to_csv("f1s.csv", index=None)

mccs = pd.DataFrame(mccs)
mccs.to_csv("mccs.csv", index=None)

voting = pd.DataFrame(voting)
voting.to_csv("voting.csv", index=None)

print(scores)
print(f1s)
print(mccs)
print(voting)


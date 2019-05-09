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

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def mode(predictions):
	return max(set(predictions), key=predictions.count)

def reproducing_sakar():
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

def griding():
	print("Gridsearching to find best parameters...")
	
	# Number of random trials
	NUM_TRIALS = 30
	
	# Load the dataset
	iris = load_iris()
	X_iris = iris.data
	y_iris = iris.target
	
	# Set up possible values of parameters to optmizer over
	p_grid = {"C"     : [0.1, 1, 10, 100],
	          "gamma" : [.001, .01, .1]}
	
	# We will use a Support Vector Classifier with "rbf" kernel
	svm = SVM(kernel="rbf")
	
	# Arrays to score scores
	non_nested_scores = np.zeros(NUM_TRIALS)
	nested_scores = np.zeros(NUM_TRIALS)
	
	# Loop for each trial
	for i in range(NUM_TRIALS):
		# Choose cross-validation techniques for the inner and outer loops,
		# independently of the dataset.
		# E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
		inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
		outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
		
		# Non_nested parameter search and scoring
		clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
		clf.fit(X_iris, y_iris)
		print(clf.best_params_, ">", clf.best_score_)
		non_nested_scores[i] = clf.best_score_
		
		# Nested CV with parameter optimization
		nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
		nested_scores[i] = nested_score.mean()
	
	score_difference = non_nested_scores - nested_scores
	
	print("Average difference of {0:6f} with std. dev. of {1:6f}.".format(
	       score_difference.mean(), score_difference.std()))
	
	# Plot scores on each trial for nested and non-nested CV
	plt.figure()
	plt.subplot(211)
	non_nested_scores_line, = plt.plot(non_nested_scores, color="r")
	nested_line, = plt.plot(nested_scores, color="b")
	plt.ylabel("score", fontsize="14")
	plt.legend([non_nested_scores_line, nested_line],
	           ["Non-Nested CV", "Nested CV"],
	           bbox_to_anchor=(0, .4, .5, 0))
	plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
	          x=.5, y=1.1, fontsize="15")
	
	# Plot bar chart of the difference
	difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
	plt.xlabel("Individual Trial #")
	plt.legend([difference_plot],
	           ["Non-Nested CV - Nested CV Score"],
	           bbox_to_anchor=(0, 1, .8, 0))
	plt.ylabel("score difference", fontsize="14")
	
	plt.show()
	
if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	
	griding()
	
	

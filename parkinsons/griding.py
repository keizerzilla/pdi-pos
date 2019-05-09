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
	

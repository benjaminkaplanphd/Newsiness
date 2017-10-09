import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, learning_curve, validation_curve, train_test_split


def plot_roc_curves(classifiers, X, y, extra_name=""):
	"""Produce ROC curves for models given feature set X and labels y
	"""
	plt.figure()
	for classifier in classifiers:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=41)

		y_score = classifiers[classifier].fit(X_train, y_train).predict_proba(X_test)[:, 1]

		fpr, tpr, _ = roc_curve(y_test, y_score)
		roc_auc = auc(fpr, tpr)

		plt.plot(fpr, tpr, lw=2, label='%s (area = %0.2f)' % (classifier, roc_auc))
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curves')
	plt.legend(loc="lower right")
	if extra_name != "":
		plt.savefig('figures/SVM_ROC_curve'+extra_name+'.png',bbox_inches="tight")
	#plt.show()


def plot_learning_curves(classifiers, X, y, extra_name=""):
	"""Produce learning curves for models given feature set X and labels y
	"""

	sample_space = np.linspace(10, len(X) * 0.8, 10, dtype='int')

	if len(classifiers) == 1:
		plt.figure(figsize=(9, 6))
	else:
		plt.figure(figsize=(9 * len(classifiers), 6))
	n_subfigure = 10 * len(classifiers) + 100
	i_subfigure = 1
	for classifier in classifiers:
		plt.subplot(n_subfigure + i_subfigure)
		i_subfigure += 1

		train_size, train_score, valid_score = learning_curve(
			estimator=classifiers[classifier],
			X=np.array(X),
			y=np.array(y),
			train_sizes=sample_space,
			cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
			scoring='accuracy')

		train_avg = np.array([np.mean(scores) for scores in train_score])
		train_std = np.array([np.std(scores) for scores in train_score])
		valid_avg = np.array([np.mean(scores) for scores in valid_score])
		valid_std = np.array([np.std(scores) for scores in valid_score])
		
		plt.fill_between(train_size, train_avg - train_std,
				train_avg + train_std, alpha=0.1,
				color="r")
		plt.fill_between(train_size, valid_avg - valid_std,
				valid_avg + valid_std, alpha=0.1, color="g")
		
		plt.plot(train_size, train_avg, label='Training Sample', color='r')
		plt.plot(train_size, valid_avg, label='Cross-validation Sample', color='g')
		plt.xlabel('Sample Size', size="xx-large")
		plt.ylabel('Classification Accuracy', size="xx-large")
		plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], (0.8, 0.85, 0.9, 0.95, 1.0), size="x-large")
		plt.xticks([0, 50, 100, 150, 200], (0, 50, 100, 150, 200), size="x-large")
		plt.legend(loc="lower right", fontsize="xx-large")
		plt.title("Learning Curves", size="xx-large")

	if extra_name != "":
		plt.savefig('figures/SVM_learning_curve'+extra_name+'.png',bbox_inches="tight")
	#plt.show()

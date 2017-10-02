import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, learning_curve, validation_curve, train_test_split


def plot_roc_curves(classifiers, X, y):
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
	plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pickle
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
		plt.title("Learning Curves ("+classifier+")", size="xx-large")

	if extra_name != "":
		plt.savefig('figures/SVM_learning_curve'+extra_name+'.png',bbox_inches="tight")
	#plt.show()


def plot_histogram(distance, source, path):
	plt.close("all")
	hist = pickle.load(file=open('inputs/histogram_inputs.pickle', 'rb'))
	plt.hist(hist, bins=40, range=(-5,5), alpha=0.5, label=source, color="blue", normed=True)
	plt.hist((distance), weights=[0.25], color="red")
	plt.xticks([-3, -1, 0, 1, 3], ('Newsy', '-1', '0', '1', 'Not-Newsy'), size="xx-large")
	plt.yticks([])
	plt.savefig(path, bbox_inches='tight', transparent=True)


def plot_corpus_histogram(classifier, features, path):
	plt.close("all")
	plt.figure(figsize=(9, 18))
	plt.subplot(311)
	distanceTRN = []
	for source, lab in zip(('reuters', 'associated-press', 'nytcontributors'),
						('Reuters', 'The AP', 'Opinions')):
	    out = classifier.decision_function(features[source])
	    distanceTRN = np.concatenate((distanceTRN,out),axis=0)
	    plt.hist(out, bins=40, range=(-5, 5), alpha=0.5, label=lab)
	plt.legend(loc="upper left", fontsize="xx-large")
	plt.ylabel("Number of Articles", size="xx-large")
	plt.xlabel("Distance to Hyperplane", size="xx-large")
	plt.yticks([])
	plt.xticks( [-3,-1,0,1,3], ('Newsy',1,'0',1,'Not-Newsy'), size="xx-large")

	distanceNYT = classifier.decision_function(features['nyteditorials'])
	distanceBLM = classifier.decision_function(features['bloomberg'])
	distanceBLM[:] = [d if abs(d) < 5 else np.sign(d) * 4.9 for d in distanceBLM]

	plt.subplot(312)
	plt.hist(distanceTRN,bins=40,range=(-5, 5), alpha=0.25,label='Training Set',color="blue", normed=True)
	plt.hist(distanceNYT,bins=40,range=(-5, 5), alpha=0.5,label='NYT Editorials',color="red",normed=True)
	plt.xticks( [-3,-1,0,1,3], ('Newsy',1,'0',1,'Not-Newsy'),size="xx-large")
	plt.yticks( [] )
	plt.legend(fontsize="xx-large")
	plt.subplot(313)
	plt.hist(distanceTRN,bins=40,range=(-5, 5), alpha=0.25,color="blue", normed=True)
	plt.hist(distanceBLM,bins=40,range=(-5, 5), alpha=0.5,label='Bloomberg',color="darkviolet",normed=True)
	plt.xticks( [-3,-1,0,1,3], ('Newsy',1,'0',1,'Not-Newsy'), size="xx-large")
	plt.yticks( [] )
	plt.legend(fontsize="xx-large")
	plt.savefig(path, bbox_inches="tight")
	plt.show()

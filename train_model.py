#!/usr/bin/env python
import psycopg2
import pandas as pd
import numpy as np
import sys, getopt
from newsiness_modules import feature_extraction as nm_fe
from newsiness_modules import news_classification as nm_nc
from newsiness_modules import newsiness_plots as nm_np
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier

redoTraining = False
doRF = False
doLR = False
nDim = 300
modelName = "temp_model"
username = 'kaplan'


def prepareModel():
	global username
	global modelName
	print "Retrieving word2vector..."
	w2v = nm_fe.retrieve_word2vector(nDim)

	con = psycopg2.connect(database='NewsinessDB',
						   host='localhost',
						   user=username,
						   password='12345')

	sql = "SELECT source_name,url,source_text,time_found FROM articles"

	df = pd.read_sql_query(sql, con)

	print "Retrieving tf-idf weights..."
	w2w = nm_fe.retrieve_word2weight(np.array(df["source_text"]))

	print "Extracting features..."

	f, t = nm_fe.extract_training_features(df, w2v, w2w, True, modelName)

	X, y = nm_nc.prepare_training_inputs(features=f, ground_truth=t)

	return X, y


def getDefaultXy():
	global modelName
	f, t = nm_nc.load_training_inputs(modelName)

	X, y = nm_nc.prepare_training_inputs(features=f, ground_truth=t)

	return X, y


def getModels():
	global doRF
	global doLR
	classifiers = {}
	classifiers['SVM'] = svm.SVC(kernel='linear', probability=True)
	if doRF:
		classifiers['Random Forest'] = RandomForestClassifier(n_estimators=10, random_state=42)
	if doLR:
		classifiers['Logistic Regression'] = linear_model.LogisticRegression()
	return classifiers


def main(argv):
	global redoTraining
	global modelName
	global doRF
	global doLR
	global nDim
	global username
	try:
		opts, args = getopt.getopt(argv, "h", ["help", "redo", "doRF", "doLR", "nDim=", "modelName=", "DBuser="])
	except getopt.GetoptError:
		print 'test.py [-h --help --redo=<bool> --doRF=<bool> --doLR=<bool> --nDim=<int> --modelName=<str> --DBuser=<str>]'
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print """
				test.py [-h --help --redo=<bool> --doRF=<bool> --doLR=<bool> --nDim=<int> --dir=<str>]'
					-h --help     : Displays this help menu'
					--redo : Redo the training.  Requires access to the local SQL database with news articles.'
					--doRF : Adds a Random Forest model to the list of classifiers
					--doLR : Adds a Logistic Regression model to the list of classifiers
					--nDim=<int>  : Specify the number of features to use in the word2vec model.  Only implemented when used with --redo=True
					--modelName=<str>   : (Default = 'temp_model') Specify a name to use when saving/retrieving the features, ground truth and model
					--DBuser=<str>: (Default = 'kaplan') Specify the username needed to acces NewsinessDB
			"""
			sys.exit()
		elif opt == "--redo":
			redoTraining = True
		elif opt == "--doRF":
			doRF = True
		elif opt == "--doLR":
			doLR = True
		elif opt == "--nDim":
			nDim = arg
		elif opt == "--modelName":
			modelName = arg
		elif opt == "--DBuser":
			username = arg

	if redoTraining:
		X, y = prepareModel()
	else:
		X, y = getDefaultXy()

	models = getModels()

	nm_np.plot_roc_curves(models, X, y, modelName)

	nm_np.plot_learning_curves(models, X, y, modelName)


if __name__ == "__main__":
	main(sys.argv[1:])

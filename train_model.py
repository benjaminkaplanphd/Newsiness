import psycopg2
import pandas as pd
import numpy as np
from newsiness_modules import feature_extraction as nm_fe
from newsiness_modules import news_classification as nm_nc
from newsiness_modules import newsiness_plots as nm_np
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier


print "Retrieving word2vector..."
w2v = nm_fe.retrieve_word2vector(300)

con = psycopg2.connect(database='NewsinessDB',
					   host='localhost',
					   user='kaplan',
					   password='12345')

sql = "SELECT source_name,url,source_text FROM articles"

df = pd.read_sql_query(sql, con)

print "Retrieving tf-idf weights..."
w2w = nm_fe.retrieve_word2weight(np.array(df["source_text"]))

print "Extracting features..."

f, t = nm_fe.extract_training_features(df, w2v, w2w, False)

X, y = nm_nc.prepare_training_inputs(features=f, ground_truth=t)

classifiers = {}
classifiers['SVM'] = svm.SVC(kernel='linear',probability=True)
classifiers['Random Forest'] = RandomForestClassifier(n_estimators=10, random_state=42)
classifiers['Logistic Regression'] = linear_model.LogisticRegression()

#for c in classifiers:
#	classifiers[c].fit(X, y)

nm_np.plot_roc_curves(classifiers, X, y)

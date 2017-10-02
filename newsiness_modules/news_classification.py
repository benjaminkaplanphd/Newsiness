import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_training_inputs(extra_name):
    name = extra_name + '.pickle'
    features = pickle.load(open('features' + name, 'r'))
    ground_truth = pickle.load(open('g_truth' + name, 'r'))
    return features, ground_truth


def prepare_training_inputs(sources=('reuters',
                                     'associated-press',
                                     'nytcontributors'),
                            features={},
                            ground_truth={}):
    X = []
    y = []

    for source in sources:
        for f, t in zip(features[source], ground_truth[source]):
            X.append(f)
            y.append(t)

    le = LabelEncoder()
    y_enc = le.fit_transform(np.array(y).ravel())
    return X, y_enc


def getResult(vector=None, classifier=None):
    if vector is None or classifier is None:
        print "Missing vector or classifier"
    return classifier.predict(vector)[0] == 0


def getNewsProb(vector=None, classifier=None):
    if vector is None or classifier is None:
        print "Missing vector or classifier"
    return classifier.predict_proba(vector)[0][0]
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_training_inputs(extra_name):
    """ Loads (previously computed) features and associated groud truth labels
        :type extra_name: str
        :rtype: Dict{str:List[List[float]]}, Dict{str:List[str]}
    """
    name = extra_name + '.pickle'
    features = pickle.load(open('inputs/features' + name, 'r'))
    ground_truth = pickle.load(open('inputs/g_truth' + name, 'r'))
    return features, ground_truth


def prepare_training_inputs(sources=('reuters',
                                     'associated-press',
                                     'nytcontributors'),
                            features={},
                            ground_truth={}):
    """ Prepares a matrix X and (0,1) labels y
        from the features and ground_truth dictionaries
        :type sources: tuple(str)
        :type features: Dict{str:List[List[float]]}
        :type ground_truth: Dict{str:List[str]}
        :rtype: List[List[float]], List[float]
    """
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
    """ Returns the prediction
        for a vector with the classifier
    """
    if vector is None or classifier is None:
        print "Missing vector or classifier"
    return classifier.predict(vector)[0] == 0


def getNewsProb(vector=None, classifier=None):
    """ Returns the prediction probability
        for a vector with the classifier
    """
    if vector is None or classifier is None:
        print "Missing vector or classifier"
    return classifier.predict_proba(vector)[0][0]

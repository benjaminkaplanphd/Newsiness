import collections
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import wordcloud
import matplotlib.pyplot as plt


def get_trained_classifier(features, types):
	X = []
	y = []
	for source in ('reuters', 'associated-press', 'nytcontributors'):
		for f, t in zip(features[source], types[source]):
			X.append(f)
			y.append(t)

		le = LabelEncoder()
		y_enc = le.fit_transform(np.array(y).ravel())

	classifier = svm.SVC(kernel='linear', probability=True)
	classifier.fit(X, y_enc)
	return classifier


def get_tfidf_weight_map(texts):

	tfidf = TfidfVectorizer(analyzer=lambda x: x)
	tfidf.fit(texts)
	max_idf = max(tfidf.idf_)
	word2weight = collections.defaultdict(
		lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
	return word2weight


def get_clouds(classifier, w_vectors, s_vectors, w_nTop, s_nTop, uid=None):

	if uid is None:
		uid = 0
	w_results = {}
	for word in w_vectors.keys():
		w_results[word] = classifier.decision_function(
			w_vectors[word].reshape(1, -1))[0]

	w_f_names = np.array(w_results.keys())
	w_res = np.array(w_results.values()).ravel()
	w_top_pos = np.argsort(w_res)[-w_nTop:]
	w_top_neg = np.argsort(w_res)[:w_nTop]

	w_freq_pos = {}
	w_freq_neg = {}

	for i in w_top_pos:
		w_freq_pos[w_f_names[i]] = abs(w_res[i])

	for i in w_top_neg:
		w_freq_neg[w_f_names[i]] = abs(w_res[i])

	wc = wordcloud.WordCloud(max_words=20, width=250, height=300)

	wc.generate_from_frequencies(w_freq_pos)
	wc.to_file('web_app/static/images/wCloud_N%d.jpg' % uid)
	wc.generate_from_frequencies(w_freq_neg)
	wc.to_file('web_app/static/images/wCloud_O%d.jpg' % uid)

	s_results = {}
	for sent in s_vectors.keys():
		s_results[sent] = classifier.decision_function(
			s_vectors[sent].reshape(1, -1))[0]

	s_f_names = np.array(s_results.keys())
	s_res = np.array(s_results.values()).ravel()
	s_top_pos = np.argsort(s_res)[-s_nTop:]
	s_top_neg = np.argsort(s_res)[:s_nTop]

	s_freq_pos = {}
	s_freq_neg = {}

	for i in s_top_pos:
		s_freq_pos[s_f_names[i]] = abs(s_res[i])

	for i in s_top_neg:
		s_freq_neg[s_f_names[i]] = abs(s_res[i])

	wc = wordcloud.WordCloud(max_words=20, width=600, height=200)

	wc.generate_from_frequencies(s_freq_pos)
	wc.to_file('web_app/static/images/sCloud_N%d.jpg' % uid)
	wc.generate_from_frequencies(s_freq_neg)
	wc.to_file('web_app/static/images/sCloud_O%d.jpg' % uid)

	return


def get_bars(classifier, w_vectors, s_vectors, w_nTop, s_nTop, uid=None):
	w_results = {}
	for word in w_vectors:
	    vector = w_vectors[word]
	    w_results[word] = classifier.decision_function(vector.reshape(1,-1))[0]
	nTop = w_nTop
	res = np.array(w_results.values()).ravel()
	top_pos = np.argsort(res)[-nTop:]
	top_neg = np.argsort(res)[:nTop]
	top = np.hstack([top_neg, top_pos])

	norm = np.max(res)
	res[:] = [r/norm for r in res]
	#print np.max(abs(res[top_pos])),np.max(abs(res[top_neg]))
	#res[top_pos][:] = [r/np.max(abs(res[top_pos])) for r in res[top_pos]]
	#res[top_neg][:] = [r/np.max(abs(res[top_neg])) for r in res[top_neg]]

	feature_names = np.array(w_results.keys())
	feature_names[:] = ["   "+f+"   " for f in feature_names]

	plt.figure(figsize=(15, 10))
	plt.subplot(122)
	#colors = ['red' if c < 0 else 'blue' for c in res[top]]
	plt.barh(np.arange(nTop), res[top_pos], color='blue', alpha = 0.25)
	plt.yticks(np.arange(0, 1 + 10), feature_names[top_pos], rotation=0, ha='left', size='xx-large', weight='bold')
	#plt.xlabel('Relative Score',size='xx-large')
	plt.xticks([1,0],('LEAST',''), size='xx-large')
	plt.title("%d Least Newsy Words"%nTop,size=25)
	sp = plt.subplot(121)
	#colors = ['red' if c < 0 else 'blue' for c in res[top]]
	plt.barh(np.arange(nTop), list(reversed(res[top_neg])), color='red', alpha = 0.25)
	plt.yticks(np.arange(0, 1 + nTop), list(reversed(feature_names[top_neg])), rotation=0, ha='right', size='xx-large', weight='bold')
	sp.yaxis.tick_right()
	#plt.xlabel('Relative Score',size='xx-large')
	plt.xticks([-1,0],('MOST',''), size='xx-large')
	plt.title("%d Most Newsy Words"%nTop,size=25)
	plt.savefig('web_app/static/images/article_word_rankings_%d.png' % uid, bbox_inches='tight')

	s_results = {}
	for s in s_vectors:
	    l=len(s.split(' '))
	    if 'said' in s or l<2:
	        continue
	    if s_vectors[s].shape != (300,):
	        continue
	    else:
	        s_results[s] = classifier.decision_function(s_vectors[s].reshape(1,-1))[0]

	nTop = s_nTop
	res = np.array(s_results.values()).ravel()
	top_pos = np.argsort(res)[-nTop:]
	top_neg = np.argsort(res)[:nTop]
	top = np.hstack([top_neg, top_pos])

	norm = np.max(res)
	res[:] = [r/norm for r in res]

	feature_names = np.array(s_results.keys())
	# feature_names[:] = [f.replace('\xe2\x80\x99','\'').replace('\xe2\x80\x94','-') for f in feature_names]
	feature_names[:] = ["   "+f+"   " for f in feature_names]

	plt.figure(figsize=(20, 5))
	plt.barh(np.arange(nTop), res[top_pos], color='blue', alpha = 0.25)
	plt.yticks(np.arange(0, 1 + nTop), feature_names[top_pos], rotation=0, ha='left', size='large', weight='bold')
	#plt.xlabel('Relative Score',size='xx-large')
	plt.xticks([1.05*max(res[top_pos]),0],("LEAST",""),size="xx-large")
	plt.title("Least Newsy Sentences",size=25)
	plt.savefig('web_app/static/images/article_notnewsy_sentences_%d.png' % uid, bbox_inches='tight')

	plt.figure(figsize=(20, 5))
	sp = plt.subplot(111)
	plt.barh(np.arange(nTop), list(reversed(res[top_neg])), color='red', alpha = 0.25)
	plt.yticks(np.arange(0, 1 + nTop), list(reversed(feature_names[top_neg])), rotation=0, ha='right', size='large', weight='bold')
	sp.yaxis.tick_right()
	plt.xticks([1.05*min(res[top_neg]),0],("MOST",""),size="xx-large")
	plt.title("Most Newsy Sentences",size=25)
	plt.savefig('web_app/static/images/article_newsy_sentences_%d.png' % uid, bbox_inches='tight')

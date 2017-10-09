from flask import request, render_template, redirect, url_for, session
from web_app import app
import matplotlib.pyplot as plt
from newsiness_modules import feature_extraction as nm_fe
from newsiness_modules import text_utils as nm_tu
from newsiness_modules import newsiness_utils as nutils
from newsiness_modules import word2vector_utils as w2v_utils
from newsiness_modules import scrape_text
import psycopg2
import pickle
import numpy as np

print "preparing the model..."

classifier = pickle.load(file=open('inputs/classifier.pickle', 'rb'))
corpus = pickle.load(file=open('inputs/texts.pickle', 'r'))

word2weight = nutils.get_tfidf_weight_map(corpus)

con = psycopg2.connect(database='WORD2VECTOR',
					   host='localhost',
					   user='kaplan',
					   password='12345')

print 'ready!'

@app.route('/')
@app.route('/index')
def index():
	if 'uid' not in session or session["uid"] == 0:
		uid = np.random.randint(100, size=1)[0]
		session["uid"] = uid
	return render_template("index.html")


@app.route('/demo')
def demo():
	session["uid"] = 0
	return render_template("index.html")


@app.route('/newsiness_analysis', methods=['GET'])
def newsiness_analysistr():
	if 'uid' not in session:
		return redirect(url_for('index'))

	if session['uid'] != 0:
		url = request.args.get('article_url')
		body = request.args.get('article_body').encode('utf-8')

		if url == '' and body == '':
			return render_template("index.html",
								   message='No input was given')

		if url != '':
			source = ''
			if 'reuters' in url:
				source = 'reuters'
			elif 'apnews' in url:
				source = 'associated-press'
			elif 'nyt' in url:
				source = 'nyt'
			if source == '':
				return render_template("index.html",
									   message='Invalid URL given')
			body = scrape_text.scrape_text(source=source, url=url).encode('utf-8')
		else:
			source = 'User Input'

		if body == '':
			return render_template("index.html",
								   message='Invalid URL given')

		full_vector = nm_fe.text_to_vector(body, con, word2weight)
		result = classifier.predict(full_vector.reshape(1, -1))[0]
		prob = float(classifier.predict_proba(full_vector.reshape(1, -1))[0][0])
		prob *= 100.
		distance = classifier.decision_function(full_vector.reshape(1, -1))[0]
		if result == 'news':
			the_result = 'NEWSY'
		else:
			the_result = 'NOT NEWSY'
			prob = 100. - prob
		prob = "%.2f" % prob

		sentences = nm_tu.text_to_sentences(body)

		sentence_vectors = {}
		word_vectors = {}
		s_out = []
		for s in sentences:
			s_for_vec = nm_tu.get_clean_sentence(s)
			if s_for_vec is None:
				result = "Quote"
				s_out.append(dict(result=result, sentence=s, prob='-', distance='-'))
				continue

			doc_vec = nm_fe.sentence_to_vector(s_for_vec, con, word2weight)
			if len(doc_vec.reshape(1, -1)[0]) != 300:
				continue
			result = classifier.predict(doc_vec.reshape(1, -1))[0]
			s_prob = float(classifier.predict_proba(doc_vec.reshape(1, -1))[0][0])
			s_prob *= 100.
			s_distance = "%.2f" % classifier.decision_function(doc_vec.reshape(1, -1))[0]
			if result == 'news':
				result = 'Newsy'
			else:
				result = "Not"
				s_prob = 100. - s_prob
			s_prob = "%.2f" % s_prob
			for w in nm_tu.text_to_wordlist(s_for_vec):
				w_vector = w2v_utils.get_vector(con, w)
				if w_vector is None:
					continue
				if w not in word_vectors:
					word_vectors[w] = w_vector * word2weight[w]
			s_out.append(dict(result=result, sentence=s, prob=s_prob, distance=s_distance))
			sentence_vectors[s] = doc_vec

		nutils.get_clouds(w_vectors=word_vectors,
						  s_vectors=sentence_vectors,
						  w_nTop=20,
						  s_nTop=5,
						  classifier=classifier,
						  uid=session['uid'])

		nutils.get_bars(w_vectors=word_vectors,
						s_vectors=sentence_vectors,
						w_nTop=10,
						s_nTop=5,
						classifier=classifier,
						uid=session['uid'])
		plt.close("all")
		hist = pickle.load(file=open('inputs/histogram_inputs.pickle', 'rb'))
		plt.hist(hist, bins=40, range=(-5,5), alpha=0.5, label=source, color="blue", normed=True)
		plt.hist((distance), weights=[0.25], color="red")
		plt.xticks([-3, -1, 0, 1, 3], ('Newsy', '-1', '0', '1', 'Not-Newsy'),size="xx-large")
		plt.yticks([])
		plt.savefig('web_app/static/images/histogram%d.png' % session['uid'], bbox_inches='tight', transparent=True)

	else:
		demo_results = pickle.load(file=open('inputs/demo_results.pickle', 'rb'))
		the_result = demo_results["full_result"]
		prob = demo_results["full_prob"]
		distance = demo_results["full_distance"]
		s_out = demo_results["sentences"]

	return render_template("newsiness_analysis.html",
						   full_result=the_result,
						   full_prob=prob,
						   full_distance=distance,
						   sentences=s_out,
						   sessionID=session['uid'])

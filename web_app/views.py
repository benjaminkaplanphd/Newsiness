from flask import request, render_template, redirect, url_for, session
from web_app import app
import matplotlib.pyplot as plt
from newsiness_modules import feature_extraction as nm_fe
from newsiness_modules import text_utils as nm_tu
from newsiness_modules import newsiness_utils as nutils
from newsiness_modules import word2vector_utils as w2v_utils
from newsiness_modules import scrape_text
from newsiness_modules import newsiness_plots as nm_np
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
			elif 'washingtonpost' in url:
				source = 'the-washington-post'
			elif 'bloomberg' in url:
				source = 'bloomberg'
			elif 'usatoday' in url:
				source = 'usatoday'
			if source == '':
				return render_template("index.html",
									   message='Invalid URL given')
			body = scrape_text.scrape_text(source=source, url=url).encode('utf-8')
		else:
			source = 'User Input'

		if body == '':
			return render_template("index.html",
								   message='Invalid URL given')

		the_result, prob, distance, s_out = nm_nu.process_article(body, con, word2weight, classifier, session['uid'], 'web_app/static/images/')
		nm_np.plot_histogram(distance, source, 'web_app/static/images/histogram%d.png' % session['uid'])

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

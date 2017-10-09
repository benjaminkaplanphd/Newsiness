#!/usr/bin/env python
from newsiness_modules import feature_extraction as nm_fe
from newsiness_modules import newsiness_utils as nutils
from newsiness_modules import scrape_text
from newsiness_modules import newsiness_plots as nm_np
from newsiness_modules import news_classification as nm_nc
import sys, getopt
import pickle


def main(argv):
	URL = ''
	fileName = ''
	corpus = False
	try:
		opts, args = getopt.getopt(argv, "h", ["help", "url=", "file=", "corpus"])
	except getopt.GetoptError:
		print 'analyze.py [-h --help --url=<str> --file=<str> --corpus]'
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print """
				analyze.py [-h --help --url=<str> --file=<str> --corpus]'
					-h --help     : Displays this help menu'
					--url=<str> : Specifies the URL to the article to analyze.
								  Supported domains are nytimes.com, reuters.com, apnews.com, bloomberg.com, washingtonpost.com, and usatoday.com
					--file=<str>: Specifies a local file with text to analyze
					** Note: Either url or file MUST be specified, and not BOTH
			"""
			sys.exit()
		elif opt == "--file":
			fileName = arg
		elif opt == "--url":
			URL = arg
		elif opt == '--corpus':
			corpus = True

		if fileName == '' and URL == '' and not corpus:
			print """
					** Note: Either url, file, or corpus MUST be specified, and not more than one
			"""
			sys.exit()
		if fileName != '' and URL != '':
			print """
					** Note: Either url or file must be specified, and not BOTH
			"""
			sys.exit()
		if corpus and (URL != '' or fileName != ''):
			print """
					** Note: Either url, file or corpus must be specified, and not MORE THAN ONE
			"""
			sys.exit()

		if URL != '':
			source = ''
			if 'reuters' in URL:
				source = 'reuters'
			elif 'apnews' in URL:
				source = 'associated-press'
			elif 'nyt' in URL:
				source = 'nyt'
			elif 'washingtonpost' in URL:
				source = 'the-washington-post'
			elif 'bloomberg' in URL:
				source = 'bloomberg'
			elif 'usatoday' in URL:
				source = 'usatoday'
			if source == '':
				print """
					** Note: Invalid URL!
				"""
				sys.exit()
			body = scrape_text.scrape_text(source=source, url=URL).encode('utf-8')
		elif fileName != '':
			with open(fileName, 'r') as content_file:
				body = content_file.read().encode('utf-8')

		classifier = pickle.load(file=open('inputs/classifier.pickle', 'rb'))

		if corpus:
			f, t = nm_nc.load_training_inputs('temp_model')
			X, y = nm_nc.prepare_training_inputs(features=f, ground_truth=t)
			classifier.fit(X, y)
			nm_np.plot_corpus_histogram(classifier, f, 'figures/distance_after_training_temp.png')
		else:
			corpus = pickle.load(file=open('inputs/texts.pickle', 'r'))

			word2weight = nutils.get_tfidf_weight_map(corpus)

			w2v = nm_fe.retrieve_word2vector()

			_, _, distance, _ = nutils.process_article(body, None, word2weight, classifier, 0, 'figures/', w2v)
			nm_np.plot_histogram(distance, source, 'figures/histogram0.png')

if __name__ == "__main__":
	main(sys.argv[1:])

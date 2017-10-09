import numpy as np
import pickle
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from newsiness_modules import scrape_text as st
from newsiness_modules import text_utils as tu
from newsiness_modules import word2vector_utils as w2v_utils


def retrieve_word2vector(dim=300, directory="../glove"):
    with open("%s/glove.6B.%dd.txt" % (directory, dim), "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}
    return w2v


def retrieve_word2weight(texts=None):
    if texts is None:
        print "Need full corpus to setup Tf-idf weights"
        return None
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(texts)
    max_idf = max(tfidf.idf_)
    word2weight = collections.defaultdict(lambda: max_idf,
                                          [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    return word2weight


def text_to_vector(text=None,
                   con=None,
                   word2weight=None,
                   w2v=None):
    out_vector = []
    text = tu.text_link_cleanup(text)
    text = tu.text_quote_cleanup(text)
    sentences = tu.split_into_sentences(text)
    sentences = tu.sentences_quote_cleanup(sentences)
    for s in sentences:
        s_for_vec = tu.get_clean_sentence(s)
        if s_for_vec is None:
            continue
        for w in tu.text_to_wordlist(s_for_vec):
            if con is not None:
                w_vector = w2v_utils.get_vector(con, w)
            elif w in w2v:
                w_vector = w2v[w]
            if w_vector is None:
                continue
            out_vector.append(w_vector * word2weight[w])
    return np.mean(np.array(out_vector), axis=0)


def sentence_to_vector(sentence=None,
                       con=None,
                       word2weight=None,
                       w2v=None):
    out_vector = []
    sentence = tu.get_clean_sentence(sentence)
    if sentence is None:
        return out_vector
    for w in tu.text_to_wordlist(sentence):
        if con is not None:
            w_vector = w2v_utils.get_vector(con, w)
        elif w in w2v:
            w_vector = w2v[w]
        if w_vector is None:
            continue
        out_vector.append(w_vector * word2weight[w])
    return np.mean(np.array(out_vector), axis=0)


def extract_training_features(df=None,
                              word2vector=None,
                              word2weight=None,
                              save=True,
                              extra_name=''):
    if df is None:
        print "Cannot extract features, if no data is provided!"
        return None

    if word2vector is None:
        word2vector = retrieve_word2vector()

    urls = df["url"]
    sources = df["source_name"]
    texts = df["source_text"]
    times = df["time_found"]

    if word2weight is None:
        word2weight = retrieve_word2weight(texts)

    ground_truth = {}
    features = {}
    word_list = []
    sentence_list = []

    known_sources = ['reuters', 'associated-press',
                     'nyteditorials','nytcontributors',
                     'usa-today','bloomberg',
                     'the-new-york-times',
                     'the-washington-post']
    training_sources = ['reuters','associated-press']
    for source in known_sources:
        features[source] = []
        ground_truth[source] = []

    nByURL = 0
    nProcessed = 0
    for url, source, text, time in zip(urls, sources, texts, times):
        if source not in known_sources:
            continue
        if source in training_sources and time>20170920093859:
            continue
        if len(text) < 5:
            text = st.scrape_text(source, url)
            nByURL += 1
        if len(text) == 0:
            print "Failed with ", source, url
            break
        text = tu.text_quote_cleanup(text)
        sentences = tu.split_into_sentences(text)
        sentences = tu.sentences_quote_cleanup(sentences)
        if len(sentences) <= 5:
            continue
        a_vec = []
        for s in sentences:
            if len(s) < 2:
                continue
            sentence_list.append(s)
            s_for_vec = tu.get_clean_sentence(s)
            if s_for_vec is None:
                continue
            for w in tu.text_to_wordlist(s_for_vec):
                if w not in word2vector:
                    continue
                word_list.append(w)
                a_vec.append(word2vector[w] * word2weight[w])

        feature = np.mean(np.array(a_vec), axis=0)

        features[source].append(feature)
        if source == 'reuters' or source == 'associated-press':
            ground_truth[source].append('news')
        else:
            ground_truth[source].append('opan')
        if (nProcessed % 10) == 0:
            print "Processed %d articles" % nProcessed
        nProcessed += 1

    print nByURL

    set(word_list)
    set(sentence_list)

    if save:
        name = extra_name + '.pickle'
        pickle.dump(file=open('inputs/features' + name, 'w'), obj=features)
        pickle.dump(file=open('inputs/g_truth' + name, 'w'), obj=ground_truth)
        pickle.dump(file=open('inputs/w_list' + name, 'w'), obj=word_list)
        pickle.dump(file=open('inputs/s_list' + name, 'w'), obj=sentence_list)

    return features, ground_truth
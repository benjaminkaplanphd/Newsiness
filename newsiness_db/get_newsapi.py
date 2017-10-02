import requests
import time
import datetime
import pickle


def get(source):
    key = "7708b865b4154217bdb03308c0bccb2b"
    url = "https://newsapi.org/v1/articles?source="
    r = requests.get(url + source + "&sortBy=top&apiKey=" + key)
    data = r.json()
    out = []
    for art in data["articles"]:
        out.append(str(art["url"]))
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    filename = source + '_urls_' + st + '.pickle'
    with open('data/' + filename, 'w') as f:
        pickle.dump(out, f)


def get_newsapi():
    get("reuters")
    get("associated-press")
    get("bloomberg")
    get("buzzfeed")
    get("cnn")
    get("national-geographic")
    get("new-york-magazine")
    get("the-new-york-times")
    get("the-wall-street-journal")
    get("the-washington-post")
    get("usa-today")

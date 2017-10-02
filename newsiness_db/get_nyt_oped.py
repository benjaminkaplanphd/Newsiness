import requests
import time
import datetime
import pickle
from lxml import html


def get(source):
    url = "https://www.nytimes.com/section/opinion/"
    page = requests.get(url + source)
    tree = html.fromstring(page.content)
    data = tree.xpath('//a[@class="story-link"]/@href')
    data[:] = [str(x) for x in data]
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    filename = "nyt" + source + '_urls_' + st + '.pickle'
    with open('data/' + filename, 'wb') as f:
        pickle.dump(data, f)


def get_nyt_oped():
    get("editorials")
    get("contributors")

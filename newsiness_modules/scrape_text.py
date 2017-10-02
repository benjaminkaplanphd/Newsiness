import requests
from lxml import html


def scrape_reuters(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    out = ""
    for p in tree.xpath('//div[@class="ArticleBody_body_2ECha"]/p/text()'):
        out = out + p + "\n"
    return out.split('-', 1)[1]


def scrape_ap(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    out = ""
    for p in tree.xpath('//div[contains(@class,"articleBody")]/p'):
        out += "".join(p.xpath('descendant-or-self::text()'))
    if len(out.split(u"\u2014", 1)) > 1:
        out = out.split(u"\u2014", 1)[1]
    return out.split("___", 1)[0]


def scrape_nyt(url):
    page = requests.get(url)
    tree = html.fromstring(page.content.decode('utf-8'))
    out = ""
    for p in tree.xpath('//p[contains(@class,"story-body-text story-content")]'):
        out += "".join(p.xpath('descendant-or-self::text()'))
    return out


def scrape_bloomberg(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    out = ""
    for p in tree.xpath('//div[contains(@class,"body-copy")]/p'):
        out += "".join(p.xpath('descendant-or-self::text()'))
    return out


def scrape_usa_today(url):
    page = requests.get(url)
    tree = html.fromstring(page.content.decode('utf-8'))
    out = ""
    for p in tree.xpath('//p[contains(@class,"p-text")]'):
        out += "".join(p.xpath('descendant-or-self::text()'))
    return out


def scrape_the_washington_post(url):
    page = requests.get(url)
    tree = html.fromstring(page.content.decode('utf-8'))
    out = ""
    for p in tree.xpath('//article[contains(@itemprop,"articleBody")]/p'):
        out += "".join(p.xpath('descendant-or-self::text()'))
    return out


def scrape_text(source, url):
    if source == "reuters":
        return scrape_reuters(url)
    elif source == "associated-press":
        return scrape_ap(url)
    elif source[0:3] == "nyt" or source == "the-new-york-times":
        return scrape_nyt(url)
    elif source[0:3] == "usa":
        return scrape_usa_today(url)
    elif source == "bloomberg":
        return scrape_bloomberg(url)
    elif source == "the-washington-post":
        return scrape_the_washington_post(url)
    else:
        return ""

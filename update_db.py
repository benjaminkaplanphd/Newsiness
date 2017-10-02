#!/usr/bin/env python
from newsiness_db import get_newsapi
from newsiness_db import get_nyt_oped
from newsiness_db import insert_articles

get_newsapi.get_newsapi()
get_nyt_oped.get_nyt_oped()
insert_articles.insert_articles()

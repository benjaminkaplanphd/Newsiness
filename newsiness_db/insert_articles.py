import pickle
import glob
import psycopg2
from newsiness_modules import scrape_text
import time
import datetime

def insert_articles():
    files = glob.glob("data/*pickle")

    con = psycopg2.connect(
        database='NewsinessDB',
        host='localhost',
        user='kaplan',
        password='12345'
    )
    cur = con.cursor()
    sql = """INSERT INTO articles (source_name,url,time_found,source_type,source_text)
             VALUES(%s, %s, %s, %s, %s)
             ON CONFLICT (url) DO NOTHING RETURNING articleID;"""

    with open("articlesDB_lastmodified", "r") as f:
        lastmodified = f.readline()
    for file in files:
        count = 0
        lines = 0
        source, _, file_time = file.split("/")[1].split(".")[0].split("_")
        print "Trying %s (%s)..." % (file, source)
        #if file_time < lastmodified:
        #    continue
        if source == "reuters" or source == "associated-press":
            source_type = 'primary'
        elif source == "nyteditorials":
            source_type = 'editorial'
        elif source == "nytcontributors":
            source_type = 'opinion'
        else:
            source_type = 'unknown'
        with open(file, 'rb') as f:
            urls = pickle.load(f)
        for url in urls:
            lines += 1
            text = scrape_text.scrape_text(source, url)
            if text is None:
                continue
            if len(text)==0:
                print url, "failed"
            else: 
                cur.execute("SELECT COUNT(*) FROM articles WHERE url = '%s'" % url )
                num = cur.fetchone()[0]
                if num==1:
                    continue
                cur.execute(sql, (source, url, file_time, source_type, text))
                if cur.fetchone() is not None:
                    count += 1
        print "Added %d/%d new %s articles" % (count, lines, source)

    con.commit()
    cur.close()
    con.close()

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    with open("articlesDB_lastmodified", 'w') as f:
        f.write(st)

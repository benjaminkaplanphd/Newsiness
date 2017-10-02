# The Newsiness Initiative
## Highlighting the analysis and opinion contained in news articles
### [http://newsiness.world](http://newsiness.world)

The purpose of this project is to classify the content of news articles, distinguishing between statements of what happened (when, where, and to whom) from those that contain analysis and opinion.

A Support Vector Machine (SVM) classifying algorithm is trained on labeled articles scrapped from various websites.  Approximately 90 articles, each, were collected from Reuters and The Associated Press, to form the 'newsy' training dataset.  Approximately 180 opinion pieces from The New York Times were used to form the 'not-newsy' training dataset.  The article URLs, and content, from these, and other, sources are stored in an SQL database.

The executable python script `./update_db.py` can be used to scrape the top articles from each news source and add them to the SQL database for later use.  This step is not neccesary to use the pre-trained classifier, stored in `inputs/classifier.pickle`, but is neccesary in order to redo the training.


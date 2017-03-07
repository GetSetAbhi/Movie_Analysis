import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import cleaner
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import os
#from sample_reviews import negative_review as review
import sample_reviews as reviews


path = os.getcwd()
df = pd.DataFrame()

df = pd.read_csv(os.path.dirname(path) + '/cleaned_data.csv')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def tokenizer(text):
	tokenized = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
	return tokenized

def preprocessor(text):
	text = re.sub('<[^>]*>', '', text).lower()
	text = cleaner.expandContractions(text)
	text = re.sub('[\W]+', ' ', text.lower())
	return text


SPLIT_PERC = 0.80
total_data = df.shape[0]
split_size = int(total_data*SPLIT_PERC)
X_train = df.loc[:split_size, 'review'].values
y_train = df.loc[:split_size, 'sentiment'].values
X_test = df.loc[split_size:, 'review'].values
y_test = df.loc[split_size:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None,
	lowercase=True,
	tokenizer=tokenizer,
	preprocessor = preprocessor,
	ngram_range=(1, 1),
	use_idf = True)


clf = Pipeline( [ ('vect', tfidf), ('clf', DecisionTreeClassifier()) ])
	
	
classifier = clf.fit(X_train, y_train)

test_label = {0:'negative', 1:'positive'}
#example = ['this movie fails to charm the audience']
example = [reviews.positive_review, reviews.positive_review2, reviews.positive_review3, reviews.negative_review, reviews.negative_review2, reviews.negative_review3]
#X = example

for rev in example:
	X = []
	X.append(rev)
	print('Prediction: %s\nProbability: %.2f%%' %(test_label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
joblib.dump(clf, "tree.p", compress=1)
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from token_proc import tokenizer, preprocessor
import sample_reviews as reviews

path = os.getcwd()
df = pd.DataFrame()

df = pd.read_csv(os.path.dirname(path) + '/cleaned_data.csv')

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


clf = Pipeline( [ ('vect', tfidf), ('clf', MultinomialNB(alpha = 1.0, fit_prior = False)) ])
	
	
classifier = clf.fit(X_train, y_train)

test_label = {0:'negative', 1:'positive'}
#example = ['this movie fails to charm the audience']
#X = example
example = [reviews.positive_review, reviews.positive_review2, reviews.positive_review3, reviews.negative_review, reviews.negative_review2, reviews.negative_review3]

for rev in example:
	X = []
	X.append(rev)
	print('Prediction: %s\nProbability: %.2f%%' %(test_label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

print('Test Accuracy: %.3f' % clf.score(X_test, y_test))	
joblib.dump(clf, "naive.p", compress=1)
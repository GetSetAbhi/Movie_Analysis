import numpy as np
import pickle
from sklearn.externals import joblib
import sample_reviews as reviews
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import HashingVectorizer
import re
import cleaner

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

vect = HashingVectorizer(decode_error='ignore',
	n_features=2**21,
	preprocessor=preprocessor,
	tokenizer=tokenizer)

clf = joblib.load("nb_online.p")
test_label = {0:'negative', 1:'positive'}

example = ['this movie fails to charm the audience']
#example = [reviews.positive_review3, reviews.negative_review3]
X = vect.transform(example)
print('Prediction: %s' %(test_label[clf.predict(X)[0]]))

import numpy as np
import pickle
from sklearn.externals import joblib
import sample_reviews as reviews
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import cleaner

'''file_Name = "naive.p"
fileObject = open(file_Name,'rb') 
clf = pickle.load(fileObject)
fileObject.close()'''

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

clf = joblib.load("lgr.p")
test_label = {0:'negative', 1:'positive'}
example = ['this movie fails to charm the audience']
#example = [reviews.positive_review3, reviews.negative_review3]
X = example
print('Prediction: %s\nProbability: %.2f%%' %(test_label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

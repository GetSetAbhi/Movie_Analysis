import numpy as np 
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
import pyprind
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from token_proc import tokenizer, preprocessor
import sample_reviews as reviews

# Loads the stopwords
root_path = os.getcwd()

# reads in and returns one document at a time:
def stream_docs(path):
	with open(path, 'r') as csv:
		next(csv)
		for line in csv:
			text, label = line[:-3], int(line[-2])
			yield text, label


'''
it will take a document stream
from the stream_docs function and return a particular number of documents
specified by the size parameter:
'''
def get_minibatch(doc_stream,size):
	docs, y = [], []
	try:
		for _ in range(size):
			text, label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except Exception as e:
		 print(e)
	
	return docs, y



vect = HashingVectorizer(decode_error='ignore',
	n_features=2**21,
	preprocessor=preprocessor,
	tokenizer=tokenizer)

#Logistic Regression Classifier
clf = BernoulliNB()
doc_stream = stream_docs(path=os.path.dirname(root_path) + '/cleaned_data.csv')

pbar = pyprind.ProgBar(40)
classes = np.array([0,1])

for _ in range(40):
	X_train, Y_train = get_minibatch(doc_stream, size = 1000)
	if not X_train:
		break

	X_train = vect.transform(X_train)
	clf.partial_fit(X_train, Y_train, classes=classes)
	pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=10000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

test_label = {0:'negative', 1:'positive'}

example = [reviews.positive_review, reviews.positive_review2, reviews.positive_review3, reviews.negative_review, reviews.negative_review2, reviews.negative_review3]

for review in example:
	temp = []
	temp.append(review)
	X = vect.transform(temp)
	print('Prediction: %s\nProbability: %.2f%%' %(test_label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

joblib.dump(clf, "nb_online.p", compress=1)
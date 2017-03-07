from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from token_proc import tokenizer, preprocessor

vect = HashingVectorizer(decode_error='ignore',
	n_features=2**21,
	preprocessor=preprocessor,
	tokenizer=tokenizer)

clf = joblib.load("svm_online.p")
test_label = {0:'negative', 1:'positive'}

example = ['this movie fails to charm the audience']
#example = [reviews.positive_review3, reviews.negative_review3]
X = vect.transform(example)
print('Prediction: %s' %(test_label[clf.predict(X)[0]]))

from sklearn.externals import joblib
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from token_proc import tokenizer, preprocessor

clf = joblib.load("svm.p")
test_label = {0:'negative', 1:'positive'}

example = ['this movie fails to charm the audience']
#example = [reviews.positive_review3, reviews.negative_review3]
X = example
print('Prediction: %s' %(test_label[clf.predict(X)[0]]))

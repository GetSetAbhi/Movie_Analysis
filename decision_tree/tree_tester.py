from sklearn.externals import joblib
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from token_proc import tokenizer, preprocessor


clf = joblib.load("tree.p")
test_label = {0:'negative', 1:'positive'}

example = ['this movie fails to charm the audience']
#example = [reviews.positive_review3, reviews.negative_review3]
X = example
print('Prediction: %s\nProbability: %.2f%%' %(test_label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

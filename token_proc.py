import re
import cleaner
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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
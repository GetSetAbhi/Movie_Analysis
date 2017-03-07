import pandas as pd
import re
import cleaner


df = pd.DataFrame()

df = pd.read_csv('./movie_data.csv')

def preprocessor(text):
	text = re.sub('<[^>]*>', '', text).lower()
	text = cleaner.expandContractions(text)
	text = re.sub('[\W]+', ' ', text.lower())
	return text



df['review'] = df['review'].apply(preprocessor)
item = df.loc[0, 'review']
print(item)
df.to_csv('./cleaned_data.csv', index=False)

#print(tokenizer(item))



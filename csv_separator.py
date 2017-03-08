import pandas as pd 

df = pd.DataFrame()

positive = pd.DataFrame(columns = ['review','sentiment'])
negative = pd.DataFrame(columns = ['review','sentiment'])


df = pd.read_csv('./cleaned_data.csv')

for index,row in df.iterrows():
	if int(row['sentiment']) == 1:
		positive = positive.append({
			'review' : row['review'],
			'sentiment' : int(row['sentiment']) }, 
			ignore_index = True)
	else:
		negative = negative.append({
			'review' : row['review'],
			'sentiment' : int(row['sentiment']) }, 
			ignore_index = True)


positive.to_csv('./positive.csv', index=False)
negative.to_csv('./negative.csv', index=False)

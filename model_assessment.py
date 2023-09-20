import pandas as pd
from tqdm import tqdm
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from datetime import timedelta

df = pd.read_csv('train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

def filter_events(df):
	"""
	Filter the dataset to include only ±8h before and after each event to avoid overtraining
	on label 'clear'
	"""
	hours = 3
	event_df = df.copy()
	event_df['label'] = event_df['label'].replace(['explosion', 'build', 'recovery'], 'event')

	event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
	event_info = event_df.groupby(['label', event_identifier]).agg(
		start_time=('timestamp', 'min'),
		end_time=('timestamp', 'max'),
		duration=('timestamp', lambda x: x.max() - x.min())
	)
	
	event_info['start_time'] -= timedelta(hours=hours)
	event_info['end_time'] += timedelta(hours=hours)

	df.set_index('timestamp', inplace=True)
	result = []
	for _, row in event_info.loc['event'].iterrows():
		result.append(df.loc[row['start_time']:row['end_time']])
	return pd.concat(result).drop_duplicates()

df = filter_events(df)
df = df.head(25000)

# Create rolling windows for X, Y and Z and happen N columns for each component
window_size = 15
for component in ['X', 'Y', 'Z']:
	columns = []
	for i in range(window_size):
		columns.append(f'{component}{i}')

	for _, window in tqdm(enumerate(df[component].rolling(window=window_size))):
		if len(window) != window_size:
			continue

		df.loc[window.index.max(), columns] = window.to_list()

df = df.drop(['X', 'Y', 'Z'], axis=1)
df.dropna(inplace=True)

y = df.pop('label').to_numpy()
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.01, random_state=42, stratify=y)

# split the dataset to concentrate on explosion maybe 2h before and 2h after
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
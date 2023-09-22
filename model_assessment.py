import pandas as pd
from tqdm import tqdm
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from datetime import timedelta
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

df = pd.read_csv('train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# import plotly.express as px
# trace_X_data = px.scatter(
#     df,
#     x='timestamp',
#     # y=pruned_df['train'].apply(lambda x: x[14] if x is not None else None),
#     y="X",
#     color="label"
# )
# trace_X_data.show()

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
# df = df.head(25000)

# Create rolling windows for X, Y and Z and happen N columns for each component
window_size = 45
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

def percentage_of_day(dt):
	"""
	Express time of the day as a percentage (100 = 24 hours)
	"""
    total_seconds_in_day = 24 * 60 * 60
    seconds_since_midnight = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return seconds_since_midnight / total_seconds_in_day

df['time'] = df.index.to_series().apply(percentage_of_day)

y = df.pop('label').to_numpy()
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

dump(clf, 'model.joblib') 

# clf = LazyClassifier(verbose=0, ignore_warnings=True)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# print(models)
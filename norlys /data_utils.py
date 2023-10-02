import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import config

def read_training_dataset(solar_wind=False):
	"""
	Read the training datasets and returning a pandas DataFrame.
	if `solar_wind` is set to true, the solar wind dataset from DSCOVR will be merged.
	"""

	df = pd.read_csv(config.TRAIN_PATH)
	if solar_wind:
		sw_df = pd.read_csv(config.SOLAR_WIND_PATH)
		df = pd.merge(df, sw_df, on='timestamp')
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	return df

def filter_events(df):
	"""
	Filter the dataset to include only ±2h before and after each event to avoid overtraining
	on label 'clear'
	"""

	event_df = df.copy()
	# Replace all label classes except 'clear' by 'event' to create one structure for events
	event_df['label'] = event_df['label'].replace(config.CLASSES, 'event')

	# Group events and compute start_time, end_time and duration on these events
	event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
	event_info = event_df.groupby(['label', event_identifier]).agg(
		start_time=('timestamp', 'min'),
		end_time=('timestamp', 'max'),
		duration=('timestamp', lambda x: x.max() - x.min())
	)
	
	# Add the window offset to include a bit of 'clear' time before and after
	event_info['start_time'] -= timedelta(hours=config.EVENT_WINDOW_OFFSET)
	event_info['end_time'] += timedelta(hours=config.EVENT_WINDOW_OFFSET)

	df.set_index('timestamp', inplace=True)
	result = []
	for _, row in event_info.loc['event'].iterrows():
		result.append(df.loc[row['start_time']:row['end_time']])
	return pd.concat(result).drop_duplicates()

def get_rolling_window(df, components=['X', 'Y', 'Z']):
	"""
	Create rolling windows for X, Y and Z and happen N columns for each component
	"""

	for component in components:
		columns = []
		for i in range(config.ROLLING_WINDOW_SIZE):
			columns.append(f'{component}{i}')

		for _, window in tqdm(enumerate(df[component].rolling(window=config.ROLLING_WINDOW_SIZE))):
			if len(window) != config.ROLLING_WINDOW_SIZE: 
				continue

			df.loc[window.index.max(), columns] = window.to_list()
	
	return df.drop(components, axis=1)
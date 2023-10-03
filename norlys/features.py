from norlys.data_utils import read_training_dataset
from sklearn.ensemble import IsolationForest

# Features needed
# - deflection score for explosion label, only if >= 5 data points. percentile of deflection for 50%, 60%, 70%, 80% and 90%
# - isolation forest number of anomalies + percentile of number of anomalies in event
# - "patterns" on the magnetometre circular map for each class, e.g. arc for "build-up", covers the whole sky for "explosion"
# - general "disturbance" score based on the deflection score and the anomaly score + the class
# - mean of the value over 45 mn
#
# Features to be added to the model
# - length of build-up if one occured in the past 45 minutes
# - position of arc depending on the derivative of Z

def get_quantiles(df):
	X = df[['X']].values 
	model = IsolationForest(random_state=42)
	model.fit(X)

	df['anomaly'] = model.predict(X)

	print(df)

	event_df = df.copy().reset_index()
	event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
	event_info = event_df.groupby(['label', event_identifier]).agg(
		start_time=('timestamp', 'min'),
		end_time=('timestamp', 'max'),
		duration=('timestamp', lambda x: x.max() - x.min()),
		deflection=('X', lambda x: x.max() - x.min()),
		anomalies=('anomaly', lambda x: x.value_counts().get(-1, 0))
	)

	print(event_info)

	return event_info.loc['explosion']['deflection'].quantile([0.5, 0.6, 0.7, 0.8, 0.9]).values

historical_data = read_training_dataset()
deflection_quantiles = get_quantiles(historical_data)

# TODO: modify embedding structure by adding labels into the rolling window 

def deflection_score(embedding):
	"""
	Calculate the deflection score of the given embedding for the last event. 
	The last event must be an explosion, otherwise it will return None. 
	The score is based on a scale of 1 to 5 each assigned to a percentile starting at 50% and increasing by 10% for each level.
	"""

	event_df = embedding.copy().reset_index()
	event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
	event_info = event_df.groupby(['label', event_identifier]).agg(
		duration=('timestamp', lambda x: x.max() - x.min()),
		deflection=('X', lambda x: x.max() - x.min())
	)	
	last_event = event_info.xs(len(event_info), level=1)
	last_event = last_event.reset_index()

	if not (last_event['label'] == 'explosion').all():
		return None
	
	for i in range(len(deflection_quantiles)):
		quantile = deflection_quantiles[i]
		if last_event['deflection'].item() <= quantile:
			# Return the index to indicate a score, not a value
			return i + 1
	return 5

def anomaly_score(embedding):
	print('test')
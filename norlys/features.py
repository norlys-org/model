from norlys.data_utils import get_formatted_data

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

def get_deflection_quantiles(df):
	event_df = df.copy().reset_index()
	event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
	event_info = event_df.groupby(['label', event_identifier]).agg(
		start_time=('timestamp', 'min'),
		end_time=('timestamp', 'max'),
		duration=('timestamp', lambda x: x.max() - x.min()),
		deflection=('X', lambda x: x.max() - x.min())
	)

	return event_info.loc['explosion']['deflection'].quantile([0.5, 0.6, 0.7, 0.8, 0.9]).values

historical_data = get_formatted_data()
deflection_quantiles = get_deflection_quantiles(historical_data)

# TODO: modify embedding structure by adding labels into the rolling window 

def deflection_score(embedding):
	for i in range(len(q)):
		quantile = q[i]
		if val <= quantile:
			# Return the index to indicate a score, not a value
			return i + 1
	return 5
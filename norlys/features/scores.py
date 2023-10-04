import pandas as pd
from datetime import timedelta
from norlys.features.quantiles import historical_data, event_info, deflection_q, explosion_anomalies_q, build_anomalies_q

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


def find_matching_quantile(quantiles, value):
	for i in range(len(quantiles)):
		quantile = quantiles[i]
		if value <= quantile:
			return i + 1
	return 5

def compute_scale_helper(data, label, column, compute_feature, quantiles, event_info):
	df = data.set_index('timestamp')
	scores = pd.DataFrame()
	for _, row in event_info.loc[label].iterrows():
		points = df.loc[row['start_time']:row['end_time']]
		score = find_matching_quantile(quantiles, row[column])

		for index, _ in points.iterrows():
			time_elapsed = index - row['start_time']
			scores = scores.append({ 
				'time elapsed': time_elapsed,
				'deflection': compute_feature(points),
				'score': score
			}, ignore_index=True)
	
	return scores.groupby(['score', 'time elapsed']).mean()

def compute_duration_scales(df, event_info):
	"""
	Compute a DataFrame with two indexes, deflection and anomaly score and elapsed time with a value which is the deflection value average.
	Using this DataFrame we can estimate in real time what the score will be because the model will process real time data and thus
	cannot compute an accurate deflection score. So we estimate using the score (computed using the deflection at the end of the event)
	and the mean values at the given duration in the event.
	"""

	compute_deflection = lambda x: x['X'].max() - x['X'].min()
	deflection_scores = compute_scale_helper(df, 'explosion', 'deflection', compute_deflection, deflection_q, event_info)

	count_anomalies = lambda x: x['anomaly'].value_counts().get(-1, 0)
	anomalies_explosion_scores = compute_scale_helper(df, 'explosion', 'anomalies', count_anomalies, explosion_anomalies_q, event_info)
	anomalies_build_scores = compute_scale_helper(df, 'build', 'anomalies', count_anomalies, build_anomalies_q, event_info)

	return (deflection_scores, anomalies_explosion_scores, anomalies_build_scores)

deflection_scale, anomaly_explosion_scale, anomaly_build_scale = compute_duration_scales(historical_data, event_info)

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

	# Get the 'local' scale that represents all mean values for each score at the specified duration
	# through the event. If the specified duration is longer than the historical data, take the maximum duration's scores
	duration = last_event['duration'].item()
	while True:
		try:
			local_scale = deflection_scale.xs(duration, level=1)
			if local_scale.shape[0] > 1: 
				break
		except KeyError:
			pass
		
		duration -= timedelta(minutes=1)

	# If there is only one index no interpolation is possible so we reduce with one minute
	# until we find a scale with at least two indexes
	while local_scale.shape[0] <= 1:
		duration -= timedelta(minutes=1)
		local_scale = deflection_scale.xs(duration, level=1)

	local_scale = local_scale.reindex([1,2,3,4,5])
	local_scale['deflection'] = local_scale['deflection'].interpolate()

	return find_matching_quantile(local_scale['deflection'].values, last_event['deflection'].item())

def anomaly_score(embedding):
	print('test')
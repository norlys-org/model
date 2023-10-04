from norlys.data_utils import read_training_dataset
from sklearn.ensemble import IsolationForest
import config

isolation_forest = IsolationForest(random_state=42)

def get_quantiles(df):
	X = df[['X']].values 
	isolation_forest.fit(X)

	df['anomaly'] = isolation_forest.predict(X)

	event_df = df.copy().reset_index()
	event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
	event_info = event_df.groupby(['label', event_identifier]).agg(
		start_time=('timestamp', 'min'),
		end_time=('timestamp', 'max'),
		duration=('timestamp', lambda x: x.max() - x.min()),
		deflection=('X', lambda x: x.max() - x.min()),
		anomaly=('anomaly', lambda x: x.value_counts().get(-1, 0))
	)

	return (
		event_info,
		event_info.loc['explosion']['deflection'].quantile(config.QUANTILES).values,
		event_info.loc['explosion']['anomaly'].quantile(config.QUANTILES).values,
		event_info.loc['build']['anomaly'].quantile(config.QUANTILES).values,
	)

historical_data = read_training_dataset()
event_info, deflection_q, explosion_anomalies_q, build_anomalies_q = get_quantiles(historical_data)
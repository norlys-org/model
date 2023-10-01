import pd as pandas

def read_training_dataset(solar_wind=False):
	"""
	Read the training datasets and returning a pandas DataFrame.
	if `solar_wind` is set to true, the solar wind dataset from DSCOVR will be merged.
	"""

	df = pd.read_csv('train.csv')
	if solar_wind:
		sw_df = pd.read_csv('sw.csv')
		df = pd.merge(df, sw_df, on='timestamp')
	df['timestamp'] = pd.to_datetime(df['timestamp'])
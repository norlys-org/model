import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

df = pd.read_csv('data/train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Create rolling windows for X, Y and Z and happen N columns for each component
# window_size = 45
# components = ['X', 'Y', 'Z']
# for component in components:
# 	for _, window in tqdm(enumerate(df[component].rolling(window=window_size))):
# 		if len(window) != window_size: 
# 			continue

# 		df.loc[window.index.max(), component] = window.to_list()

event_df = df.copy()
event_df = event_df.reset_index()
event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
event_info = event_df.groupby(['label', event_identifier]).agg(
	start_time=('timestamp', 'min'),
	end_time=('timestamp', 'max'),
	duration=('timestamp', lambda x: x.max() - x.min()),
	deflection=('X', lambda x: x.max() - x.min())
)

# Deflection score based on percentiles
q = event_info.loc['explosion']['deflection'].quantile([0.5, 0.6, 0.7, 0.8, 0.9]).values
def deflection_score(val):
	for i in range(len(q)):
		quantile = q[i]
		if val <= quantile:
			# Return the index to indicate a score, not a value
			return i + 1
	return 5

# Add deflections to all rows contained in the 'explosion' events
for _, row in event_info.loc['explosion'].iterrows():
	df.loc[row['start_time']:row['end_time'], 'deflection'] = deflection_score(row['deflection']) * 50

# Isolation forest
X = df[['X']].values 
model = IsolationForest(random_state=42)
model.fit(X)
predictions = model.predict(X)

anomaly_df = df.copy()
anomaly_df['anomaly'] = predictions
anomaly_df = anomaly_df[anomaly_df['anomaly'] == -1]

# Convert labels to integers for colouring
labels = ['clear', 'build', 'explosion', 'recovery', 'energy_entry']
df['label'] = df['label'].apply(lambda x: labels.index(x)) 

# Plot
trace1 = go.Scatter(x=df.index, y=df.X, mode='markers', marker=dict(color=df['label']), name='Magnetogram X')
trace2 = go.Scatter(x=anomaly_df.index, y=anomaly_df.anomaly, mode='markers', name='X anomalies')
trace3 = go.Bar(x=df.index, y=df.deflection, name='Explosion deflections', opacity=0.8, marker=dict(color='blue'))

fig = go.Figure(data=[trace1, trace2, trace3])

fig.update_layout(
    title='Magnetogram extracted features',
    xaxis=dict(title='Time'),
    yaxis=dict(title='X (nT)')
)
fig.show()
import pandas as pd

buf = []
for year in ['2018', '2020']:
	buf.append(pd.read_csv(f'to-label-{year}-labeled.csv'))
og_df = pd.concat(buf)

# Assuming you want to aggregate the values for duplicate timestamps
# If you have a specific aggregation function, replace 'mean' with that function
df = og_df.groupby(['timestamp', 'series'])['value'].mean().unstack()

# Merge with label
df = df.merge(og_df[['timestamp', 'label']].drop_duplicates(), on='timestamp', how='left')
# df.set_index('timestamp', inplace=True)
df = df.drop_duplicates(subset='timestamp', keep='first')
df = df.drop('class', axis=1)
df['label'] = df['label'].fillna('clear')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
df.to_csv('train.csv', index=False)
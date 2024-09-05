import pandas as pd
from legacy.config import config

for station in ['TAR']:
  df = pd.read_csv(f'legacy/temp/{station}.csv')
  df['UT'] = pd.to_datetime(df['UT'])
  df.set_index('UT', inplace=True)
  df = df[~df.index.duplicated(keep='first')]
  df_resampled = df.resample('min').mean()
  df_interpolated = df_resampled.interpolate(method='linear')

  df_interpolated.to_csv(f'data/magnetograms/{station}.csv')
import json
import os
import pandas as pd
from datetime import datetime, timedelta

import requests
from legacy.config import config
from sklearn.model_selection import train_test_split
from requests_toolbelt.multipart.encoder import MultipartEncoder

def get_training_data(y_column, rw_components=[], solar_wind=False):
  return get_data(y_column, rw_components=rw_components, solar_wind=solar_wind)

def get_formatted_data(rw_components=[], solar_wind=False):
  return get_data('', rw_components=rw_components, solar_wind=solar_wind, model=False)

def get_clipped_data():
  df = read_training_dataset()
  df = filter_events(df)
  return df.head(572) # Testing arbitrary because build period ends after 486th element

def get_data(y_column, rw_components=[], solar_wind=False, model=True):
  """
  Returns the fully ready data for training using all other formatting functions.
  if model is set to True the data will be split into X and Y and test + train.
  """

  df = read_training_dataset(solar_wind=solar_wind)
  df = filter_events(df)
  df = get_rolling_window(df, components=rw_components)
  if not model:
    return df
  return training_format(df, y_column=y_column)

def read_training_dataset(solar_wind=False):
  """
  Read the training datasets and returning a pandas DataFrame.
  if `solar_wind` is set to true, the solar wind dataset from DSCOVR will be merged.
  """

  df = pd.read_csv(config['pathes']['trainPath'])
  if solar_wind:
    sw_df = pd.read_csv(config['pathes']['solarWindPath'])
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
  event_df['label'] = event_df['label'].replace(config['classes'], 'event')

  # Group events and compute start_time, end_time and duration on these events
  event_identifier = (event_df['label'] != event_df['label'].shift()).cumsum()
  event_info = event_df.groupby(['label', event_identifier]).agg(
    start_time=('timestamp', 'min'),
    end_time=('timestamp', 'max'),
    duration=('timestamp', lambda x: x.max() - x.min())
  )
  
  # Add the window offset to include a bit of 'clear' time before and after
  event_info['start_time'] -= timedelta(hours=config['eventWindowOffset'])
  event_info['end_time'] += timedelta(hours=config['eventWindowOffset'])

  df.set_index('timestamp', inplace=True)
  result = []
  for _, row in event_info.loc['event'].iterrows():
    result.append(df.loc[row['start_time']:row['end_time']])
  return pd.concat(result).drop_duplicates()

def get_rolling_window(df, components=[]):
  """
  Create rolling windows for X, Y and Z and happen N columns for each component
  """

  components += ['X', 'Y', 'Z']
  # Do not write 'label' into the components to not drop it later
  # for component in components + ['label']:
  for component in components:
    columns = []
    for i in range(config['rollingWindowSize']):
      columns.append(f'{component}{i}')

    pbar = enumerate(df[component].rolling(window=config['rollingWindowSize']))
    for _, window in pbar:
      if len(window) != config['rollingWindowSize']: 
        continue

      df.loc[window.index.max(), columns] = window.to_list()
  
  return df.drop(components, axis=1)

def percentage_of_day(dt):
  """
  Express time of the day as a percentage (100 = 24 hours)
  """
  total_seconds_in_day = 24 * 60 * 60
  seconds_since_midnight = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
  return seconds_since_midnight / total_seconds_in_day

def training_format(df, y_column):
  """
  Split the given DataFrame into X and y for testing and training the models.
  """

  df.dropna(inplace=True)
  df['time'] = df.index.to_series().apply(percentage_of_day) # Add time as a feature in the dataset

  y = df.pop(y_column).to_numpy()
  X = df
  return train_test_split(X, y, test_size=.3, random_state=42, stratify=y)


def write_to_kv(key, value):
  url = f"https://api.cloudflare.com/client/v4/accounts/{config['accountID']}/storage/kv/namespaces/{config['namespaceID']}/values/{key}"

  # Create the multipart encoder
  m = MultipartEncoder(
      fields={
          'metadata': json.dumps({
            'date': datetime.utcnow().isoformat()
          }),
          'value': value
      }
  )

  token = os.environ.get('CF_API_TOKEN')
  headers = {
      'Content-Type': m.content_type,
      'Authorization': f'Bearer {token}'
  }

  return requests.put(url, data=m, headers=headers)

def write_to_d1(value):
  url = f"https://api.cloudflare.com/client/v4/accounts/{config['accountID']}/d1/database/{config['d1ID']}/query"

  date = datetime.utcnow().strftime('%Y-%m-%d')

  token = os.environ.get('CF_API_TOKEN')
  payload = {
    'params': [date, value, value],
    'sql': 'INSERT INTO Archive (date, value) VALUES (?, ?) ON CONFLICT(date) DO UPDATE SET value = MAX(value, ?)'
  }
  headers = {
    'Authorization': f'Bearer {token}'
  }

  return requests.post(url, json=payload, headers=headers)
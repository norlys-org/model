import logging
from app.baseline import compute_long_term_baseline, get_substracted_data
from app.data_utils import get_clipped_data, get_rolling_window, get_training_data, percentage_of_day
from app.features.quantiles import compute_scores
from app.fetch import fetch_mag
from app.model import load_0m_classifier
import pandas as pd
from app.rendering import create_matrix
import numpy as np
from multiprocessing import Pool, cpu_count
from app.rendering import create_matrix
from config import config
import warnings
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

warnings.simplefilter(action='ignore', category=FutureWarning) # TODO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

clf = load_0m_classifier()

def mean_score(scores):
    """
    Calculate a weighted average score from a set of given scores.

    The function takes a dictionary of scores, each corresponding to a different metric. 
    It computes the weighted average using predefined weights for each metric. Currently, 
    all metrics have equal weights of 1. The weighted average is computed by summing the 
    products of each score and its weight, then dividing by the total sum of weights.

    Parameters:
    scores (dict): A dictionary with metric names as keys and their scores as values.

    Returns:
    float: The calculated weighted mean score.
    """

    # z deviation good ponderation
    # table = {
    #     'X_rolling_anomalies': 0, 'X_rolling_gradient': 0, 'X_deflection': 0, 'X_deviation': 1, 
    #     'Y_rolling_anomalies': 0, 'Y_rolling_gradient': 0, 'Y_deflection': 0, 'Y_deviation': 0, 
    #     'Z_rolling_anomalies': 0, 'Z_rolling_gradient': 0, 'Z_deflection': 0, 'Z_deviation': 0
    # }
    table = {
      'X_deviation': 1, 
    }

    sum = 0
    weights_sum = 0
    for key in table:
        weight = table[key]
        sum += scores[key] * weight
        weights_sum += weight

    return sum / weights_sum

def get_lon(stations):
    """
    Average all longitudes from given longitudes
    """

    sum = 0
    for station in stations:
        sum += config['magnetometres'][station]['lon']
    return sum / len(stations)

def initialize_lines_df():
    lines_df = []
    mean_lon = 0

    for line in config['magnetometreLines']:
        line_df = pd.DataFrame(index=np.arange(
            config['magnetometres'][line[-1]]['lat'], 
            config['magnetometres'][line[0]]['lat'], 
            0.01
        ))
        line_df['Z'] = np.nan
        lines_df.append(line_df)

        mean_lon += get_lon(line)

    mean_lon /= len(config['magnetometreLines'])

    return (lines_df, mean_lon)


def process_station(val):
  key, clf = val
  logging.info(f'Fetching {key}...')
  station = config['magnetometres'][key]

  logging.info(f'Retrieving month archive for {key}...')
  archive_df = pd.read_csv(f'data/month/{key}_data.csv')
  archive_df['date'] = pd.to_datetime(archive_df['date'])
  archive_df.set_index('date', inplace=True)
  archive_df = archive_df[archive_df.index >= archive_df.index.max() - pd.DateOffset(months=1)]

  logging.info(f'Fetching real-time values for {key}...')
  df = fetch_mag(station['slug'], station['source'])
  if df.empty:
      return {}

  logging.info(f'Computing baseline for {key}...')
  full_df = pd.concat([ archive_df, df ])
  full_df = full_df[~full_df.index.duplicated(keep='first')]
  full_df = full_df.resample('min').interpolate()
  full_df.dropna(inplace=True)
  full_df.sort_index(inplace=True)

  baseline = compute_long_term_baseline(key, full_df.index.min(), full_df.index.max(), full_df)
  baseline.index.names = ['date']
  result_df = full_df - baseline
  result_df.dropna(inplace=True)

  # a = go.Scatter(x=result_df.index, y=result_df['X'], mode='lines', name='X')
  # layout = go.Layout(
  #     title=f'{station}',
  #     xaxis=dict(title='Time'),
  #     yaxis=dict(title='Value')
  # )
  # fig = go.Figure(data=[a], layout=layout)
  # for val in config['deviationThresholds']:
  #     fig.add_hline(y=val, line_dash="dash", line_color="red")
  #     fig.add_hline(y=-val, line_dash="dash", line_color="red")
  # pio.show(fig)

  result_df = result_df[result_df.index >= result_df.index.max() - pd.Timedelta(minutes=45)]
  
  if result_df.empty:
    return {}

  logging.info(f'Computing scores for {key}...')
  scores = compute_scores(result_df.copy(), key)
  mean = mean_score(scores)

  # logging.info(f'Getting model prediction for {key}')
  # model_df = get_rolling_window(result_df)
  # model_df['time'] = model_df.index.to_series().apply(percentage_of_day) # Add time as a feature in the dataset
  # model_df.dropna(inplace=True)

  z = result_df['Z'].tail(1).item() 
  x = result_df['X'].tail(1).item() 

  try:
    return key, (mean, 'clear'), z, x
    # return key, (mean, clf.predict(model_df)[0]), z
  except Exception as e:
    logging.error(f'Error occurred for {key}: {str(e)}')


def interpolate_df(df):
    """
    Interpolates a dataframe and finds the minimum and maximum 'Z' index.
    """

    df = df.sort_index()
    df = df.interpolate(method='spline', order=3, s=0.)
    df['maxima'] = (df['Z'] > df['Z'].shift(1)) & (df['Z'] > df['Z'].shift(-1))
    df['minima'] = (df['Z'] < df['Z'].shift(1)) & (df['Z'] < df['Z'].shift(-1))

    extrema = df[(df['maxima'] | df['minima'])].reset_index()
    extrema['diff'] = extrema['Z'].diff().abs()
    max_diff_idx = extrema['diff'].idxmax()
    min_idx = extrema.loc[max_diff_idx - 1]['index']
    max_idx = extrema.loc[max_diff_idx]['index']

    sorted_indices = sorted([min_idx, max_idx])
    return sorted_indices[0], sorted_indices[1], df['X'].max()

def crop_oval(result, lines_df, line_lon):
    """
    Crops points in a matrix based on latitude limits defined by interpolated lines.

    This function takes a matrix of points (longitude and latitude) and uses two 
    interpolated lines to define upper and lower latitude limits. It sets the 'score' 
    of points outside these limits to 0, effectively cropping them.

    Parameters:
    result: A matrix containing points with 'lon' and 'lat' values.

    Returns:
    The matrix with scores of certain points set to 0 based on latitude limits.
    """

    matrix = create_matrix(result)

    limits_df = pd.DataFrame(index=range(-90, 40), columns=['min', 'max'])

    for i, line in enumerate(config['magnetometreLines']):
      line_lon = get_lon(line)
      if line_lon > 180:
        line_lon = line_lon - 360
      line_lon = round(line_lon)

      lat_min, lat_max, x = interpolate_df(lines_df[i])
      limits_df.loc[line_lon, 'x'] = x
      limits_df.loc[line_lon, 'min'] = lat_min
      limits_df.loc[line_lon, 'max'] = lat_max

    limits_df['min'] = limits_df['min'].astype(float)
    limits_df['max'] = limits_df['max'].astype(float)
    limits_df.interpolate(method='linear', inplace=True)
    limits_df.ffill(inplace=True)
    limits_df.bfill(inplace=True)

    # a = go.Scatter(x=limits_df.index, y=limits_df['max'], mode='lines', name='max')
    # b = go.Scatter(x=limits_df.index, y=limits_df['x'], mode='lines', name='X')
    # c = go.Scatter(x=limits_df.index, y=limits_df['min'], mode='lines', name='min')
    # layout = go.Layout(
    #     title=f'Lines',
    #     xaxis=dict(title='Latitude'),
    #     yaxis=dict(title='Value')
    # )
    # fig = go.Figure(data=[a,b,c], layout=layout)
    # pio.show(fig)
    
    for point in matrix:
      lon = point['lon']
      if lon > 180:
        lon = lon - 360
      if point['lat'] < limits_df.loc[lon, 'min']:
        point['score'] = 0
      if point['lat'] > limits_df.loc[lon, 'max']:
        point['score'] = point['score'] * 0.5
    
    return matrix


def get_matrix():
  lines_df, line_lon = initialize_lines_df()

  result = {}
  stations = []
  with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_station, [(key, clf) for key in config['magnetometres']])

    for item in results:
      if item == {}:
        continue

      key, data, z, x = item
      stations.append(key)
      result.update({ key: data })

      for i, line in enumerate(config['magnetometreLines']):
        if key in line:
          lines_df[i].loc[config['magnetometres'][key]['lat'], 'Z'] = z
          lines_df[i].loc[config['magnetometres'][key]['lat'], 'X'] = x

  for i in range(0,3):
    df = lines_df[i]
    df = df.sort_index()
    df = df.interpolate(method='spline', order=3, s=0.)
    lines_df[i] = df

  a = go.Scatter(x=lines_df[0].index, y=lines_df[0]['Z'], mode='lines', name='finnish Z')
  b = go.Scatter(x=lines_df[0].index, y=lines_df[0]['X'], mode='lines', name='finnish X')
  c = go.Scatter(x=lines_df[1].index, y=lines_df[1]['Z'], mode='lines', name='norwegian Z')
  d = go.Scatter(x=lines_df[1].index, y=lines_df[1]['X'], mode='lines', name='norwegian X')
  e = go.Scatter(x=lines_df[2].index, y=lines_df[2]['Z'], mode='lines', name='greenland Z')
  f = go.Scatter(x=lines_df[2].index, y=lines_df[2]['X'], mode='lines', name='greenland X')
  layout = go.Layout(
      title=f'Lines',
      xaxis=dict(title='Latitude'),
      yaxis=dict(title='Value')
  )
  fig = go.Figure(data=[a,b,c,d,e,f], layout=layout)
  pio.show(fig)
  print(stations)

  return crop_oval(result, lines_df, line_lon)
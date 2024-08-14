from itertools import chain
import json
import logging
from app.baseline import compute_long_term_baseline, get_substracted_data
from app.data_utils import get_clipped_data, get_rolling_window, get_training_data, percentage_of_day, write_to_kv
from app.features.quantiles import compute_scores, find_quantile_range
from app.fetch import fetch_mag
from app.model import load_0m_classifier
import pandas as pd
from app.rendering import create_matrix
import numpy as np
from multiprocessing import Pool, cpu_count
from app.rendering import create_matrix
from config import config
import warnings
import plotly.graph_objs as go
import plotly.io as pio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.basemap import Basemap
from pysecs import SECS

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

  y = result_df['Z'].tail(1).item() 
  z = result_df['Z'].tail(1).item() 
  x = result_df['X'].tail(1).item() 
  prediction = 'clear'
  # prediction = clf.predict(result_df)[0]

  try:
    return key, (mean, prediction), y, x
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
    return sorted_indices[0], sorted_indices[1], df['X'].abs().max()

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

    # for point in matrix:
    #   lon = point['lon']
    #   if lon > 180:
    #     lon = lon - 360
    #   if point['lat'] < limits_df.loc[lon, 'min']:
    #     point['score'] = 0

    #   if point['lat'] > limits_df.loc[lon, 'max']:
    #     point['score'] *= 0.5

    #   if point['lat'] >= limits_df.loc[lon, 'min'] and point['lat'] <= limits_df.loc[lon, 'max']:
    #     point['score'] *= 1.5
    
    return matrix


def get_matrix():
  lines_df, line_lon = initialize_lines_df()

  result = {}
  vector = []
  stations = []
  with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_station, [(key, clf) for key in config['magnetometres']])

    for item in results:
      if item == {}:
        continue

      key, data, z, x = item
      stations.append(key)
      result.update({ key: data })

      lon = config['magnetometres'][key]['lon']
      if lon > 180:
        lon = lon - 360

      vector.append({ 
        'lat': config['magnetometres'][key]['lat'],
        'lon': lon,
        'x': x,
        'z': z
      })

      for i, line in enumerate(config['magnetometreLines']):
        if key in line:
          lines_df[i].loc[config['magnetometres'][key]['lat'], 'Z'] = z
          lines_df[i].loc[config['magnetometres'][key]['lat'], 'X'] = x

  vector_df = pd.DataFrame(vector)

  x = vector_df['lon'].values
  y = vector_df['lat'].values
  u = vector_df['x'].values  # Eastward component
  v = vector_df['z'].values  # Other component

  plt.figure(1)
  plt.quiver(x, y, u, v)
  plt.title("Original Data")

  R_earth = 6371e3
  # SECS grid setup within the range of input data
  lat, lon, r = np.meshgrid(np.linspace(50, 90),
                            np.linspace(-80, 40),
                            R_earth + 110000, indexing='ij')
  secs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                              lon.reshape(-1, 1),
                              r.reshape(-1, 1)))

  secs = SECS(sec_df_loc=secs_lat_lon_r)

  # Observation grid matching input data points
  obs_lat_lon_r = np.hstack((y.reshape(-1, 1),
                            x.reshape(-1, 1),
                            np.full((len(x), 1), R_earth)))

  B_obs = np.zeros((1, len(obs_lat_lon_r), 3))
  B_obs[0, :, 0] = u
  B_obs[0, :, 1] = v

  secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs, epsilon=0.1)
  # lat_pred, lon_pred, r_pred = np.meshgrid(np.linspace(50, 85, 100),
  #                                         np.linspace(-80, 40, 200),
  #                                         R_earth, indexing='ij')
  lat_pred, lon_pred, r_pred = np.meshgrid(np.linspace(50, 85, 25),
                                          np.linspace(-80, 40, 50),
                                          R_earth, indexing='ij')
  pred_lat_lon_r = np.hstack((lat_pred.reshape(-1, 1),
                              lon_pred.reshape(-1, 1),
                              r_pred.reshape(-1, 1)))
  B_pred = secs.predict(pred_lat_lon_r)

  # Ensure B_pred has the correct shape
  if B_pred.ndim == 2:
      B_pred = B_pred[np.newaxis, ...]

  # Prepare for plotting
  u_pred = B_pred[0, :, 0].reshape(lat_pred.shape)
  v_pred = B_pred[0, :, 1].reshape(lat_pred.shape)

  # Normalize the vectors
  magnitude = np.sqrt(u_pred**2 + v_pred**2)
  u_norm = u_pred / (magnitude + 1e-10)  # Avoid division by zero
  v_norm = v_pred / (magnitude + 1e-10)

  # Plotting the normalized and interpolated vector field on a world map
  plt.figure(figsize=(12, 8))
  m = Basemap(projection='aeqd', lon_0=-10, lat_0=70, width=4000000, height=4000000)
  m.drawcoastlines()
  m.drawcountries()
  m.drawparallels(np.arange(50., 91., 10.), labels=[True, False, False, False])
  m.drawmeridians(np.arange(-80., 41., 10.), labels=[False, False, False, True])

  # Convert lat/lon to map projection coordinates
  xx_map, yy_map = m(lon_pred, lat_pred)

  # Flatten the arrays to pass to plt.quiver
  xx_map_flat = xx_map.flatten()
  yy_map_flat = yy_map.flatten()
  u_norm_flat = u_norm.flatten()
  v_norm_flat = v_norm.flatten()

  # Plot the interpolated quiver plot on the map
  plt.quiver(xx_map_flat, yy_map_flat, u_norm_flat, v_norm_flat, u_pred.flatten(), scale=50, cmap=plt.cm.seismic, clim=(-500, 500))
  plt.title("Normalized and Interpolated Data on World Map")
  
  # Adding a colorbar that represents the eastward component intensity
  plt.colorbar(label='Eastward Component Intensity')
  plt.show()

  score = [find_quantile_range(config['deviationThresholds'], abs(x)) for x in u_pred.flatten()]

  result_v = [{'lon': lon_pred.flatten()[i], 'lat': lat_pred.flatten()[i], 'x': u_pred.flatten()[i], 'z': v_pred.flatten()[i]} for i in range(len(score))]
  result = [{'lon': lon_pred.flatten()[i], 'lat': lat_pred.flatten()[i], 'score': score[i], 'status': 'clear'} for i in range(len(score))]

  print(write_to_kv('vectors', json.dumps(result_v)))
 
  return result, max(score)

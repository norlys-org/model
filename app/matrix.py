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
        'X_rolling_anomalies': 1, 'X_rolling_gradient': 1, 'X_deflection': 10, 'X_mean': 20, 
        'Y_rolling_anomalies': 1, 'Y_rolling_gradient': 1, 'Y_deflection': 10, 'Y_mean': 20, 
        'Z_rolling_anomalies': 1, 'Z_rolling_gradient': 1, 'Z_deflection': 10, 'Z_mean': 20
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

lines = [
    ['NAL', 'LYR', 'BJN', 'SOR', 'MAS', 'MUO', 'PEL', 'RAN', 'OUJ', 'HAN', 'NUR', 'TAR'],
    ['NAL', 'LYR', 'BJN', 'SOR', 'JCK', 'DON', 'RVK', 'DOB', 'SOL', 'KAR']
]
def initialize_lines_df():
    lines_df = []
    mean_lon = 0

    for line in lines:
        line_df = pd.DataFrame(index=np.arange(
            config['magnetometres'][line[-1]]['lat'], 
            config['magnetometres'][line[0]]['lat'], 
            0.01
        ))
        line_df['Z'] = np.nan
        lines_df.append(line_df)

        mean_lon += get_lon(line)

    mean_lon /= len(lines)

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
  if full_df.isna().any().any():
      full_df = full_df.interpolate()
  full_df.sort_index(inplace=True)
  
  baseline = compute_long_term_baseline(key, full_df.index.min(), full_df.index.max(), full_df)
  baseline.index.names = ['date']
  result_df = full_df - baseline
  result_df.dropna(inplace=True)
  result_df = result_df[result_df.index >= result_df.index.max() - pd.Timedelta(minutes=45)]

  logging.info(f'Computing scores for {key}...')
  scores = compute_scores(result_df.copy(), key)
  mean = mean_score(scores)

  logging.info(f'Getting model prediction for {key}')
  model_df = get_rolling_window(result_df)
  model_df['time'] = model_df.index.to_series().apply(percentage_of_day) # Add time as a feature in the dataset
  model_df.dropna(inplace=True)
  
  z = result_df['Z'].tail(1).item() 

  try:
    return key, (mean, clf.predict(model_df)[0]), z
  except Exception as e:
    logging.error(f'Error occurred for {key}: {str(e)}')


def interpolate_df(df):
    """
    Interpolates a dataframe and finds the minimum and maximum 'Z' index.
    """

    interpolated_df = df.interpolate(method='cubic')
    return interpolated_df['Z'].idxmin(), interpolated_df['Z'].idxmax()

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

    # Refactored repeated code
    lat1_min, lat1_max = interpolate_df(lines_df[0])
    lat2_min, lat2_max = interpolate_df(lines_df[1])

    for point in matrix:
        # Streamlined conditional logic
        if (point['lon'] > line_lon and (point['lat'] < lat2_min or point['lat'] > lat2_max)) or \
           (point['lon'] <= line_lon and (point['lat'] < lat1_min or point['lat'] > lat1_max)):
            point['score'] = 0
    
    return matrix


def get_matrix():
  lines_df, line_lon = initialize_lines_df()

  result = {}
  # with Pool(processes=cpu_count()) as pool:
  #   results = pool.map(process_station, [(key, clf) for key in config['magnetometres']])
  for key in config['magnetometres']:
    key, data, z = process_station((key, clf))
    result.update({ key: data })
    def set_z_val(line_df):
        line_df.loc[config['magnetometres'][key]['lat'], 'Z'] = z

    if key in lines[0]:
        set_z_val(lines_df[0])
    if key in lines[1]:
        set_z_val(lines_df[1])

  return crop_oval(result, lines_df, line_lon)
from app.baseline import get_substracted_data
from app.features.features import apply_features, get_features_column_list
import config
import json
import logging
from multiprocessing import Pool, cpu_count


def find_quantile_range(quantiles, value):
    """
    Given the array of computed quantiles of a specific component and a value it outputs the index, called score
    """
    
    for i in range(len(quantiles)):
        quantile = quantiles[i]
        if value <= quantile:
            return i + 1
    return 9

def compute_scores(df, station):
    # TODO refactor
    df = apply_features(df)
    values = {}
    for component in ['X', 'Y', 'Z']:
        for slug in get_features_column_list(component):
            values[slug] = df[slug].iloc[-1]

    result = {}
    with open(config.QUANTILES_PATH, 'r') as file:
        quantiles_data = json.load(file)
        for key in quantiles_data[station]:
            quantiles = quantiles_data[station][key]
            value = values[key]
            result[key] = find_quantile_range(quantiles, value)
    
    return result

def compute_quantiles(df):
    df = apply_features(df)

    result = {}
    for component in ['X', 'Y', 'Z']:
      for slug in get_features_column_list(component):
          result[slug] = df[slug].quantile(config.QUANTILES).tolist()

    return result

def process_station(station):
  logging.info(f'Computing and substracting baseline for {station}') 
  df = get_substracted_data(station)
  df.dropna(inplace=True)

  logging.info(f'Computing quantiles for {station}') 
  return {station: compute_quantiles(df)}

def save_quantiles():
    result = {}

    with Pool(processes=cpu_count()) as pool:
      results = pool.map(process_station, config.STATIONS)
      for item in results:
        result.update(item)

    with open(config.QUANTILES_PATH, 'w') as fp:
        json.dump(result, fp)
from norlys.baseline import get_substracted_data
from norlys.features import features
from sklearn.ensemble import IsolationForest
import config
import json
import pandas as pd
import logging


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
    values = compute_quantiles(df, False)

    result = {}
    with open(config.QUANTILES_PATH, 'r') as file:
        data = json.load(file)
        for key in data[station]:
            quantiles = data[station][key]
            value = values[key]
            result[key] = find_quantile_range(quantiles, value)
    
    return result


def compute_quantiles(df, quantiles=True):
    for component in ['X', 'Y', 'Z']:
        for feature_slug in features:
            df[f'{component}_{feature_slug}'] = features[feature_slug](df, component)
    
    df.dropna(inplace=True)

    result = {}
    for component in ['X', 'Y', 'Z']:
      for slug in [f'{component}_{slug}' for slug in features].append(component):
        if quantiles:
          result[slug] = df[slug].quantile(config.QUANTILES).tolist()
        else:
          result[slug] = df[slug].iloc[-1]

    return result

import concurrent.futures

def save_quantiles():
    result = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the processing function to each station in parallel
        results = executor.map(process_station, config.STATIONS)
        # Iterate through the results and store them in the result dictionary
        for station, quantiles in zip(config.STATIONS, results):
            result[station] = quantiles

    logging.info(f'Saving computed quantiles') 
    with open(config.QUANTILES_PATH, 'w') as fp:
        json.dump(result, fp)

def process_station(station):
    logging.info(f'Computing and substracting baseline for {station}') 
    df = get_substracted_data(station)
    df.dropna(inplace=True)

    logging.info(f'Computing quantiles for {station}') 
    return compute_quantiles(df)

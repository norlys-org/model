from norlys.baseline import get_substracted_data
from norlys.features import apply_features, get_features_column_list
from sklearn.ensemble import IsolationForest
import config
import json
import logging
import concurrent.futures

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

# TODO remove that quantiles parameter and split into two functions
def compute_quantiles(df):
    df = apply_features(df)

    result = {}
    for component in ['X', 'Y', 'Z']:
      for slug in get_features_column_list(component):
          result[slug] = df[slug].quantile(config.QUANTILES).tolist()

    return result

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

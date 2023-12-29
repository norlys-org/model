from norlys.baseline import get_substracted_data
from sklearn.ensemble import IsolationForest
import config
import json
import pandas as pd
import logging

QUANTILES_PATH = 'data/quantiles.json'

def find_quantile_range(quantiles, value):
    for i in range(len(quantiles)):
        quantile = quantiles[i]
        if value <= quantile:
            return i + 1
    return 9

def get_scores(df, station):
    values = get_quantiles(df, False)

    result = {}
    with open(QUANTILES_PATH, 'r') as file:
        data = json.load(file)
        for key in data[station]:
            quantiles = data[station][key]
            value = values[key]
            result[key] = find_quantile_range(quantiles, value)
    
    return result


def get_quantiles(df, quantiles=True):
    for component in ['X', 'Y', 'Z']:
        # Anomalies with Isolation forest
        isolation_forest = IsolationForest(random_state=42)
        X = df[[component]].values 
        isolation_forest.fit(X)

        df[f'{component}_anomaly'] = isolation_forest.predict(X)
        df[f'{component}_anomalies'] = (df[f'{component}_anomaly'] == -1).astype(int).rolling('15T').sum()

        # Gradient over the last 15 minutes
        df[f'{component}_gradient'] = df[component].diff().rolling('15T').mean()
        # Deflection over the past 45 minutes
        df[f'{component}_deflection'] = df[component].rolling('45T').apply(lambda x: x.max() - x.min())
    
    df.dropna(inplace=True)

    result = {}
    for component in ['X', 'Y', 'Z']:
        def get_result(slug):
            if quantiles:
                return df[f'{component}{slug}'].quantile(config.QUANTILES).tolist()
            
            return df[f'{component}{slug}'].iloc[-1]

        result[f'{component}_anomalies'] = get_result('_anomalies')
        result[f'{component}_gradient'] = get_result('_gradient')
        result[f'{component}_deflection'] = get_result('_deflection')
        result[f'{component}'] = get_result('')

    return result

def save_quantiles():
    result = {}
    for station in config.STATIONS:
        logging.info(f'Computing quantiles for {station}') 
        df = get_substracted_data(station)
        df.dropna(inplace=True)
        result[station] = get_quantiles(df)

    with open(QUANTILES_PATH, 'w') as fp:
        json.dump(result, fp)
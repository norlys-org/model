import logging
from norlys.baseline import compute_long_term_baseline, get_substracted_data
from norlys.data_utils import get_clipped_data, get_rolling_window, get_training_data, percentage_of_day
from norlys.features.quantiles import get_scores
from norlys.fetch import fetch_mag
from norlys.model import load_0m_classifier
import pandas as pd
import config
from norlys.rendering import create_matrix
import plotly.graph_objects as go
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

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
        'X_anomalies': 1, 'Y_anomalies': 1, 'Z_anomalies': 1,
        'X_gradient': 1, 'Y_gradient': 1, 'Z_gradient': 1,
        'X_deflection': 1, 'Y_deflection': 1, 'Z_deflection': 1,
        'X': 1, 'Y': 1, 'Z': 1,
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
        sum += config.STATIONS[station]['lon']
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
            config.STATIONS[line[-1]]['lat'], 
            config.STATIONS[line[0]]['lat'], 
            0.01
        ))
        line_df['Z'] = np.nan
        lines_df.append(line_df)

        mean_lon += get_lon(line)

    mean_lon /= len(lines)

    return (lines_df, mean_lon)

clf = load_0m_classifier()
result = {}
lines_df, line_lon = initialize_lines_df()

for key in ['NAL']:
    logging.info(f'Fetching {key}...')
    station = config.STATIONS[key]

    logging.info(f'Retrieving month archive for {key}...')
    archive_df = pd.read_csv(f'data/month/{key}_data.csv')
    archive_df['date'] = pd.to_datetime(archive_df['date'])
    archive_df.set_index('date', inplace=True)
    archive_df = archive_df[archive_df.index >= archive_df.index.max() - pd.DateOffset(months=1)]

    logging.info(f'Fetching real-time values for {key}...')
    df = fetch_mag(station['slug'], station['source'])
    if df.empty:
        continue

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

    # fig = go.Figure(data=[
    #     go.Scatter(x=result_df.index, y=result_df.X, mode='lines', name='Magnetogram X')
    # ])
    # fig.update_layout(
    #     title='Magnetogram extracted features',
    #     xaxis=dict(title='Time'),
    #     yaxis=dict(title='X (nT)')
    # )
    # fig.show()


    logging.info(f'Computing scores for {key}...')
    scores = get_scores(result_df.copy(), key)
    print(scores)
    mean = mean_score(scores)

    logging.info(f'Getting model prediction for {key}')
    model_df = get_rolling_window(result_df)
    model_df['time'] = model_df.index.to_series().apply(percentage_of_day) # Add time as a feature in the dataset
    model_df.dropna(inplace=True)
    
    z = result_df['Z'].tail(1).item() 
    def set_z_val(line_df):
        line_df.loc[config.STATIONS[key]['lat'], 'Z'] = z

    if key in lines[0]:
        set_z_val(lines_df[0])
    if key in lines[1]:
        set_z_val(lines_df[1])

    result[key] = (mean, clf.predict(model_df)[0])
    print(clf.predict(model_df)[0])

def interpolate_df(df):
    """
    Interpolates a dataframe and finds the minimum and maximum 'Z' index.
    """

    interpolated_df = df.interpolate(method='cubic')
    return interpolated_df['Z'].idxmin(), interpolated_df['Z'].idxmax()

def crop_oval(result):
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

    # for point in matrix:
    #     # Streamlined conditional logic
    #     if (point['lon'] > line_lon and (point['lat'] < lat2_min or point['lat'] > lat2_max)) or \
    #        (point['lon'] <= line_lon and (point['lat'] < lat1_min or point['lat'] > lat1_max)):
    #         point['score'] = 0
    
    return matrix

matrix = crop_oval(result)

from flask import Flask, jsonify
from flask_cors import CORS
from norlys.rendering import create_matrix
import random
import config

app = Flask(__name__)
CORS(app)

@app.route('/map', methods=['GET'])
def get_map():
	return matrix

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
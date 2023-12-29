import logging
from norlys.baseline import compute_long_term_baseline, get_substracted_data
from norlys.data_utils import get_clipped_data, get_rolling_window, get_training_data, percentage_of_day
from norlys.features.quantiles import get_scores
from norlys.fetch import fetch_mag
from norlys.model import load_0m_classifier
import pandas as pd
import config
import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def mean_score(scores):
    table = {
        'X_anomalies': 1,
        'Y_anomalies': 1,
        'Z_anomalies': 1,
        'X_gradient': 1,
        'Y_gradient': 1,
        'Z_gradient': 1,
        'X_deflection': 1,
        'Y_deflection': 1,
        'Z_deflection': 1,
        'X': 1,
        'Y': 1,
        'Z': 1,
    }

    sum = 0
    weights_sum = 0
    for key in table:
        weight = table[key]
        sum += scores[key] * weight
        weights_sum += weight

    return sum / weights_sum

get_substracted_data('TRO')

# clf = load_0m_classifier()
# result = {}
# for key in config.STATIONS:
#     logging.info(f'Fetching {key}...')
#     station = config.STATIONS[key]

#     # logging.info(f'Retrieving month archive for {key}...')
#     # archive_df = pd.read_csv(f'data/month/{key}_data.csv')
#     # archive_df['date'] = pd.to_datetime(archive_df['date'])
#     # archive_df.set_index('date', inplace=True)
#     # archive_df = archive_df[archive_df.index >= archive_df.index.max() - pd.DateOffset(months=1)]

#     logging.info(f'Fetching real-time values for {key}...')
#     df = fetch_mag(station['slug'], station['source'])

#     logging.info(f'Computing baseline for {key}...')
#     # full_df = pd.concat([ archive_df, df ])
#     full_df = df
#     if full_df.isna().any().any():
#         full_df = full_df.interpolate()
    
#     baseline = compute_long_term_baseline(key, full_df.index.min(), full_df.index.max(), full_df)
#     baseline.index.names = ['date']
#     result_df = full_df - baseline
#     result_df.dropna(inplace=True)
#     # result_df = result_df[result_df.index >= result_df.index.max() - pd.Timedelta(minutes=45)]
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=result_df.index, y=result_df['X'], mode='lines', name='DataFrame 1'))
#     fig.add_trace(go.Scatter(x=full_df.index, y=full_df['X'], mode='lines', name='DataFrame 2'))
#     fig.show()

#     logging.info(f'Computing scores for {key}...')
#     scores = get_scores(result_df.copy(), key)
#     mean = mean_score(scores)
#     print(mean)

#     logging.info(f'Getting model prediction for {key}')
#     model_df = get_rolling_window(result_df)
#     model_df['time'] = model_df.index.to_series().apply(percentage_of_day) # Add time as a feature in the dataset
#     model_df.dropna(inplace=True)

#     result[key] = (mean, clf.predict(model_df)[0])

# print(result)

# clf = load_0m_classifier()

# df = df.tail(45)
# df = get_rolling_window(df)
# df['time'] = df.index.to_series().apply(percentage_of_day) # Add time as a feature in the dataset
# df.pop('label')
# df.dropna(inplace=True)
# print(df.columns.tolist())
# # print(clf.predict(df))
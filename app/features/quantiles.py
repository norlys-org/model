from app.baseline import get_substracted_data
from app.features.features import apply_features, get_features_column_list
from config import config
import json
import logging
from multiprocessing import Pool, cpu_count
import plotly.graph_objs as go
import plotly.io as pio


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
    df = apply_features(df)
    result = {}
    result['X_deviation'] = find_quantile_range(config['deviationThresholds'], df['X_deviation'].iloc[-1])

    return result

def compute_quantiles(df):
    df = apply_features(df)

    result = {}
    for component in ['X', 'Y', 'Z']:
      for slug in get_features_column_list(component):
          result[slug] = df[slug].quantile(config['quantiles']).tolist()

    return result

def process_station(station):
  logging.info(f'Computing and substracting baseline for {station}') 
  df = get_substracted_data(station)
  df.dropna(inplace=True)
  df = df[~df.index.duplicated(keep='first')]

  a = go.Scatter(x=df.index, y=df['X'], mode='lines', name='X')
  # # b= go.Scatter(x=model_df.index, y=model_df['X'], mode='lines', name='Interpolated Data')
  # constant_lines = []
  # for i, value in enumerate(q):
  #     constant_line = go.Scatter(
  #         x=[df.index.max()],
  #         y=[value],
  #         mode='lines',
  #         name=f'X deviation {i}',
  #         line=dict(dash='dash')  # Optional: make the line dashed
  #     )
  #     constant_lines.append(constant_line)

  # Create the layout for the plot
  layout = go.Layout(
      title=f'{station}',
      xaxis=dict(title='Time'),
      yaxis=dict(title='Value')
  )
  fig = go.Figure(data=[a], layout=layout)
  pio.show(fig)

  logging.info(f'Computing quantiles for {station}') 
  return {station: compute_quantiles(df)}

def save_quantiles():
    result = {}

    with Pool(processes=cpu_count()) as pool:
      results = pool.map(process_station, config['magnetometres'])
      for item in results:
        result.update(item)

    # with open(config['pathes']['quantilesPath'], 'w') as fp:
    #     json.dump(result, fp)
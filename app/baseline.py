from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from suncalc import get_position
import math
from config import config
import plotly.graph_objs as go
import plotly.io as pio

def read_and_format(station):
    """
    Read station magnetometre's data and perform preliminary formatting on the data.
    """

    df = pd.read_csv(f'data/magnetograms/{station}_data.csv')
    df = df[(df['X'] != 999999) & (df['Y'] != 999999) & (df['Z'] != 999999)]

    df['X'] = df['X'] / 10
    df['Y'] = df['Y'] / 10
    df['Z'] = df['Z'] / 10

    df['UT'] = pd.to_datetime(df['UT'])
    df.set_index('UT', inplace=True)
    df = df.sort_index()

    df = df.head(100000)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.drop('StationId', axis=1)

    return df

def compute_long_term_baseline(station, start, end, df):
    """
    Compile data with daily median and templates based on quietest day of time range to 
    generate the long term baseline
    """

    df_daily_median = df.between_time('12:00', '12:00').resample('D').median()
    disturbed_days, quiet_days = compute_quietest_and_disturbed_days(station, start, end, df)

    df_daily_median = df_daily_median.drop(disturbed_days)
    df_daily_median = pd.DataFrame(index=pd.date_range(start, end, freq='min')).join(df_daily_median, how='left').interpolate(method='time')

    templates = []
    for quiet_day in quiet_days:
      df_quiet_day = df.loc[str(quiet_day)] - df_daily_median.loc[str(quiet_day)]
      templates.append(template(df_quiet_day))

    baseline = df_daily_median + interpolate_diurnal_baseline(start, end, templates, quiet_days)

    trace_df = go.Scatter(x=df.index, y=df['X'], mode='lines', name='Original Data')
    trace_baseline = go.Scatter(x=baseline.index, y=baseline['X'], mode='lines', name='Baseline')

    # Create the layout for the plot
    layout = go.Layout(
        title='Data, Baseline, and Difference',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Value')
    )
    fig = go.Figure(data=[trace_df, trace_baseline], layout=layout)
    pio.show(fig)

    return df_daily_median

def template(df, num_frequency_components=7):
    """
    Generate a template curve based on the quietest day's magnetic data following the paper's instructions.
    """

    result = {'X': [], 'Y': [], 'Z': []}

    # Calculate the FFT for each component and reconstruct the signal
    for component in ['X', 'Y', 'Z']:
        # Calculate FFT and take the first 7 harmonics as per paper's suggestion
        y = df[component].values
        y_fft = np.fft.fft(y)[:num_frequency_components]
        retained_fft_values = y_fft[:num_frequency_components]

        n = len(y)
        t_d = np.arange(n)
        
        template = np.zeros(n)
        for h in range(num_frequency_components):
            X_h = retained_fft_values[h]
            template += np.abs(X_h) * np.cos(2 * np.pi * h * t_d / 1440 + np.angle(X_h))
        
        result[component] = template / n

    result_df = pd.DataFrame(result, index=df.index)
    return result_df

def interpolate_diurnal_baseline(start, end, templates, quiet_days):
    """
    Interpolate between templates to create a diurnal baseline.
    
    Parameters:
    templates: list
    quiet_days: list
        A list of quiet days.
    
    Returns:
    pd.DataFrame
        A DataFrame representing the diurnal baseline.
    """

    if start.date() not in quiet_days:
      quiet_days.insert(0, start)
      templates.insert(0, templates[0])
    if end.date() not in quiet_days:
      quiet_days.append(end)
      templates.append(templates[-1])

    dfs = []

    for i in range(len(quiet_days) - 1):
      t1 = pd.Timestamp(quiet_days[i]) + pd.Timedelta(hours=12)
      t2 = pd.Timestamp(quiet_days[i + 1]) + pd.Timedelta(hours=12)
      baseline = pd.DataFrame(index=pd.date_range(t1, t2, freq='min'), columns=['X', 'Y', 'Z'])

      T1 = templates[i]
      T2 = templates[i+1]

      for component in ['X', 'Y', 'Z']:
        def interpolate(row):
          t = int(row.name.timestamp())
          t_d = int(t / 60 % 1440)
          return T1[component].iloc[t_d] + T2[component].iloc[t_d] - T1[component].iloc[t_d] * ((t - t1.timestamp()) / (t2.timestamp() - t1.timestamp()))
        baseline[component] = baseline.apply(interpolate, axis=1)

      dfs.append(baseline)
      
    return pd.concat(dfs)

def compute_quietest_and_disturbed_days(station, start, end, df):
    """
    Compute the array of σ_hmax see 2.3 Manual. 
    Analyze magnetic data to identify the quietest and most disturbed days within a specified date range.

    The function iterates through each day in the given date range and calculates the standard deviation of 
    the 'X' and 'Z' components of the magnetic data for each hour. A day is marked as disturbed if the median 
    of the hourly standard deviations in the 'X' component exceeds a predefined threshold specific to the station.
    
    The quietest day is determined based on the day with the lowest maximum sum of standard deviations ('h_max') 
    in both 'X' and 'Z' components. This day signifies the least variation in magnetic data, indicating minimal 
    magnetic disturbance.

    Parameters:
        station (str): The name of the magnetic station.
        start (str): The start date for analysis in 'YYYY-MM-DD' format.
        end (str): The end date for analysis in 'YYYY-MM-DD' format.
        df (DataFrame): A DataFrame containing magnetic data for the station.

    Returns:
        tuple: A tuple containing a list of disturbed days and the quietest day within the given date range.
    """

    h_max = pd.DataFrame(columns=['h_max'])
    disturbed_days = []
    quiet_days = []

    for day in pd.date_range(start, end, freq='D'):
        df_day = df.loc[str(day.date())]
        if df_day.empty:
            continue
        
        std_devs = pd.DataFrame(columns=['X', 'Z'])
        for hour in range(24):
            df_hour = df_day.between_time(f'{hour}:00', f'{hour}:59').copy()

            new_data = {}
            for component in ['X', 'Z']:
                if len(df_hour) < 2:
                    continue

                X = np.arange(len(df_hour)).reshape(-1, 1)
                Y = df_hour[component]
                model = LinearRegression().fit(X, Y)
                std_dev = np.sqrt(mean_squared_error(Y, model.predict(X)))
                new_data[component] = std_dev

            std_devs = pd.concat([std_devs, pd.DataFrame(new_data, index=[hour])])
        
        std_devs.dropna(inplace=True)

        h_max = pd.concat([h_max, pd.DataFrame(
            { 'h_max': max(std_devs['X'] + std_devs['Z']) },
            index=[day]
        )])

        if np.median(std_devs['X'].to_numpy()) > config['magnetometres'][station]['X']:
            disturbed_days.append(day.date())

    h_max['month'] = h_max.index.to_period('M')
    for _, group in h_max.groupby('month'):
        quiet_day = group['h_max'].idxmin()
        if len(df.loc[quiet_day:quiet_day + pd.Timedelta(hours=23, minutes=59)]) == 1440:
          quiet_days.append(quiet_day.date())

    return disturbed_days, quiet_days

def is_daytime(latitude, longitude, timestamp):
    """
    Calculate if the sun for the given coordinates and at the given timestamp is 12 degrees or more below
    the horizon
    """

    return get_position(timestamp, longitude, latitude)['altitude'] * 180/math.pi > -12

def get_substracted_data(station):
    df = read_and_format(station)

    start = df.index.min()
    end = df.index.max()
    baseline = compute_long_term_baseline(station, start, end, df)    

    return df[start:end] - baseline[start:end]
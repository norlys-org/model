from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from suncalc import get_position
import math
import config

def read_and_format(station):
    """
    Read station magnetometre's data and perform preliminary formatting on the data.
    """

    df = pd.read_csv(f'data/magnetograms/{station}_data.csv')
    df = df[(df['X'] != 999999) & (df['Y'] != 999999) & (df['Z'] != 999999)]

    df['X'] = df['X'] / 10
    df['Y'] = df['Y'] / 10
    df['Z'] = df['Z'] / 10

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.drop('StationId', axis=1)

    df['UT'] = pd.to_datetime(df['UT'])
    df.set_index('UT', inplace=True)

    return df

def compute_long_term_baseline(station, start, end, df):
    """
    Compile data with daily median and templates based on quietest day of time range to 
    generate the long term baseline
    """

    df_daily_median = df.resample('D').median()
    disturbed_days, quietest_day = compute_quietest_and_disturbed_days(station, start, end, df)

    baseline = df_daily_median.drop(disturbed_days).resample('T').mean().interpolate(method='linear')
    if quietest_day == '':
        return baseline

    tp = template(df, baseline, quietest_day)

    return tp

def template(df, baseline, quietest_day):
    """
    Compute template curve and extrapolate it over the year, based on Fast Fourier Transform harmonics.
    """

    start = quietest_day
    end = start + timedelta(days=1)

    baseline_resampled = baseline.loc[str(start) : str(end)]
    df_resampled = df.loc[str(start) : str(end)]
    residual = df_resampled - baseline_resampled

    data = {'UT': [f"{quietest_day + pd.Timedelta(minutes=i)}" for i in range(1440)], 'X': [], 'Y': [], 'Z': []}
    for component in ['X', 'Y', 'Z']:
        frequencies = np.fft.fft(residual[component])[:10]
        for i in range(0, 1440):
            sum_v = 0
            for h in range(0, 10): 
                amplitude = abs(frequencies[h])
                phase = np.angle(frequencies[h])
                sum_v += amplitude * np.cos(2 * np.pi * h * i * 60 / 86400 + phase)
            data[component].append(sum_v)

    tp = pd.DataFrame(data)
    tp['UT'] = pd.to_datetime(tp['UT'])
    tp.set_index('UT', inplace=True)

    num_days = 365
    repeated_data = [tp.copy() for _ in range(num_days)]
    for i in range(num_days):
        repeated_data[i].index = tp.index + pd.Timedelta(days=i)

    result = pd.concat(repeated_data)
    for component in ['X', 'Y', 'Z']:
        result[component] = result[component] / baseline[component] + baseline[component]
    return result

def compute_quietest_and_disturbed_days(station, start, end, df):
    """
    Compute the array of σ_hmax see 2.3 Manual. 
    Start and end date required of the format YYYY-MM-DD
    """

    h_max = pd.DataFrame(columns=['h_max'])
    disturbed_days = []

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
 
        h_max = pd.concat([h_max, pd.DataFrame(
            { 'h_max' :max(std_devs['X'] + std_devs['Z']) },
            index=[day]
        )])

        if np.median(std_devs['X'].to_numpy()) > config.STATIONS[station]['X']:
            disturbed_days.append(day.date())
 
    if len(h_max['h_max']) == 0:
        quietest_day = ''
    else:
        quietest_day = h_max['h_max'].idxmin()

    return disturbed_days, quietest_day

def is_daytime(latitude, longitude, timestamp):
    """
    Calculate if the sun for the given coordinates and at the given timestamp is 12 degrees or more below
    the horizon
    """

    return get_position(timestamp, longitude, latitude)['altitude'] * 180/math.pi > -12

def get_substracted_data(station):
    df = read_and_format(station)
    baseline = compute_long_term_baseline(station, '2018-11-01', '2018-12-30', df)

    data_hourly = df['2018-11-01':'2018-12-30']
    baseline_hourly = baseline['2018-11-01':'2018-12-30']
    return data_hourly - baseline_hourly

# df = get_substracted_data('TRO')
# df_class = read_allsky_state()
# df = df.merge(df_class[['UT', 'class']], left_on='UT', right_on='UT', how='left')
# df['class'].fillna(0, inplace=True)

# df.set_index('UT', inplace=True)
# df = df.dropna(subset=['X', 'Y', 'Z'])

# df['timestamp'] = df.index.strftime('%Y-%m-%dT%H:%M:%S.000Z')
# df = df.melt(id_vars='timestamp', var_name='series', value_name='value')
# df['label'] = None
# # df.set_index('timestamp', inplace=True)
# df[['series', 'timestamp', 'value', 'label']].to_csv('to-label.csv', index=False)

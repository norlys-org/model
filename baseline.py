from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
from scipy.interpolate import griddata
from suncalc import get_position, get_times
import math
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer

h_med_threshold_values = {
    'NAL': { 'X': 14, 'Y': 13, 'lat': 78.92, 'lon': 11.95 },
    'LYR': { 'X': 16, 'Y': 15, 'lat': 78.20, 'lon': 15.82 },
    'HOR': { 'X': 20, 'Y': 17, 'lat': 77.00, 'lon': 15.60 },
    'HOP': { 'X': 22, 'Y': 14, 'lat': 76.51, 'lon': 25.01 },
    'BJN': { 'X': 21, 'Y': 15, 'lat': 74.50, 'lon': 19.20 },
    'NOR': { 'X': 25, 'Y': 12, 'lat': 71.09, 'lon': 25.79 },
    'SOR': { 'X': 20, 'Y': 12, 'lat': 70.54, 'lon': 22.22 },
    'KEV': { 'X': 18, 'Y': 9, 'lat': 69.76, 'lon': 27.01 },
    'TRO': { 'X': 20, 'Y': 11, 'lat': 69.66, 'lon': 18.94 },
    'MAS': { 'X': 18, 'Y': 11, 'lat': 69.46, 'lon': 23.70 },
    'AND': { 'X': 20, 'Y': 11, 'lat': 69.30, 'lon': 16.03 },
    'KIL': { 'X': 18, 'Y': 10, 'lat': 69.06, 'lon': 20.77 },
    'IVA': { 'X': 16, 'Y': 8, 'lat': 68.56, 'lon': 27.29 },
    'ABK': { 'X': 17, 'Y': 10, 'lat': 68.35, 'lon': 18.82 },
    # 'LEK': { 'X': 15, 'Y': 9, 'lat': 68.13, 'lon': 13.54 },
    'MUO': { 'X': 15, 'Y': 8, 'lat': 68.02, 'lon': 23.53 },
    'LOZ': { 'X': 12, 'Y': 6, 'lat': 67.97, 'lon': 35.08 },
    'KIR': { 'X': 14, 'Y': 8, 'lat': 67.84, 'lon': 20.42 },
    'SOD': { 'X': 13, 'Y': 7, 'lat': 67.37, 'lon': 26.63 },
    'PEL': { 'X': 12, 'Y': 7, 'lat': 66.90, 'lon': 24.08 },
    'DON': { 'X': 11, 'Y': 6, 'lat': 66.11, 'lon': 12.50 },
    'RVK': { 'X': 9, 'Y': 5, 'lat': 64.94, 'lon': 10.98 },
    'LYC': { 'X': 19, 'Y': 7, 'lat': 64.61, 'lon': 18.75 },
    'OUJ': { 'X': 7, 'Y': 5, 'lat': 64.52, 'lon': 27.23 },
    'MEK': { 'X': 4, 'Y': 4, 'lat': 62.77, 'lon': 30.97 },
    'HAN': { 'X': 4, 'Y': 4, 'lat': 62.25, 'lon': 26.60 },
    'DOB': { 'X': 5, 'Y': 4, 'lat': 62.07, 'lon': 9.11 },
    'SOL': { 'X': 4, 'Y': 4, 'lat': 61.08, 'lon': 4.84 },
    'NUR': { 'X': 5, 'Y': 3, 'lat': 60.50, 'lon': 24.65 },
    'UPS': { 'X': 4, 'Y': 3, 'lat': 59.90, 'lon': 17.35 },
    'KAR': { 'X': 3, 'Y': 3, 'lat': 59.21, 'lon': 5.24 },
    'TAR': { 'X': 3, 'Y': 3, 'lat': 58.26, 'lon': 26.46 },
}

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

        if np.median(std_devs['X'].to_numpy()) > h_med_threshold_values[station]['X']:
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
    baseline = compute_long_term_baseline(station, '2020-01-01', '2021-01-01', df)

    data_hourly = df['2020-01-01':'2021-01-01']
    baseline_hourly = baseline['2020-01-01':'2021-01-01']
    return data_hourly - baseline_hourly

def read_allsky_state():
    df = pd.read_csv('data/allsky-state.csv')
    df['UT'] = pd.to_datetime(df['UT'])
    df['UT'] = df['UT'].dt.round('T')

    def determine_class(row):
        classes = ['nodata', 'arc', 'discrete', 'diffuse', 'clear']
        if row['cloudy'] > 70 or row['dd'] > 70:
            return classes.index('nodata')

        for col in classes[1:]:
            if row[col] > 90:
                return classes.index(col)
        return classes.index('clear')
    
    def class_h(row):
        return determine_class(row)

    df['class'] = df.apply(class_h, axis=1)
    return df

df = get_substracted_data('TRO')
df_class = read_allsky_state()
df = df.merge(df_class[['UT', 'class']], left_on='UT', right_on='UT', how='left')
df['class'].fillna(0, inplace=True)

# df.set_index('UT', inplace=True)

# df['is_daytime'] = df.apply(lambda row: is_daytime(h_med_threshold_values['TRO']['lat'], h_med_threshold_values['TRO']['lon'], row.UT), axis=1)
# df.loc[df['is_daytime'], ['X', 'Y', 'Z']] = 0
# df.dropna(inplace=True)

# N = 15
# df['train'] = None
# for i in range(len(df)):
#     try:
#         class_val = df.loc[i, 'class']
#         if class_val is None:
#             # or math.isnan(class_val):
#             continue
#     except KeyError:
#         continue

#     values = df.loc[i-N:i, 'X']
#     # df.at[i, 'train'] = values
#     df.at[i, 'train'] = values.sum()

# pruned_df = df.dropna(subset=['train', 'class'])
# X, y = df['X'].values, df['class'].values
# pruned_df['train_viz'] = pruned_df['train'].apply(lambda x: x.sum() if not x.empty else None)
df.set_index('UT', inplace=True)
df = df.dropna(subset=['X', 'Y', 'Z'])

# .sample(frac=0.2, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(
#     # np.concatenate([df['X'].values, df['Y'].values, df['Z'].values]),
#     X,
#     y,
#     test_size=.2,
#     random_state=64,
# )

# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(X_train, X_test, y_train, y_test)
# # base_clf = GaussianNB()
# # base_clf.fit(X_train, y_train)
# # calibrated_clf = CalibratedClassifierCV(base_clf, cv="prefit")
# # calibrated_clf.fit(X_calib, y_calib)

# print(models)

# df.to_csv('test-model.csv')
# data_m = df['2018-11-06 00:00:00':'2018-11-08 23:00:00'].interpolate(limit_direction='both')
# data_mean = data_m.resample('7T').mean()
# data_ewm = data_m.ewm(alpha=0.01, adjust=False).mean()
# data_ewm_m = data_mean.ewm(alpha=0.05, adjust=False).mean()

# df['X'] = df['X'].interpolate()

# df = df['2018-10-01 00:00:00':'2018-11-08 23:00:00']
# Create traces for the original data and the median data
# trace_X_data = px.scatter(
#     df,
#     x=df.index,
#     # y=pruned_df['train'].apply(lambda x: x[14] if x is not None else None),
#     y="X",
#     color="class"
# )
# trace_X_data.show()

df['timestamp'] = df.index.strftime('%Y-%m-%dT%H:%M:%S.000Z')
df = df.melt(id_vars='timestamp', var_name='series', value_name='value')
df['label'] = None
# df.set_index('timestamp', inplace=True)
df[['series', 'timestamp', 'value', 'label']].to_csv('to-label.csv', index=False)

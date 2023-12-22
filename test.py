from norlys.data_utils import get_clipped_data, get_rolling_window, get_training_data, percentage_of_day
from norlys.features.quantiles import get_scores
from norlys.model import load_0m_classifier
import pandas as pd

df = get_clipped_data()
print(get_scores(df.copy()))

clf = load_0m_classifier()

df = df.tail(45)
df = get_rolling_window(df)
df['time'] = df.index.to_series().apply(percentage_of_day) # Add time as a feature in the dataset
df.pop('label')
df.dropna(inplace=True)
print(clf.predict(df))
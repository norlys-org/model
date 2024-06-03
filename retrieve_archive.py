"""
This file shall be run once a day ideally just before midnight so that the data of the current day 
is added to the month archive. This month archive is used to compute the baseline.
"""

import logging
from config import config
from app.fetch import fetch_mag
import os
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

last_month_date = datetime.now().replace(hour=23, minute=59, second=59) - timedelta(days=30)
for key in config['magnetometres']:
    logging.info(f'Fetching {key}')
    station = config['magnetometres'][key]
    df = fetch_mag(station['slug'], station['source'])

    file_path = f'data/month/{key}_data.csv'

    # Check if the file exists
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        existing_df.set_index('date', inplace=True)

        combined_df = pd.concat([existing_df, df]).sort_index()
    else:
        combined_df = df

    filtered_df = combined_df.loc[last_month_date:]
    filtered_df.to_csv(file_path, index=True)

current_date = datetime.now()
date_string = current_date.strftime('%Y-%m-%d')

# Write it to a text file
with open('data/archive_update_date.txt', 'w') as file:
    file.write(date_string)
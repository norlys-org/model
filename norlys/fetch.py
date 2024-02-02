import requests
import os
from datetime import datetime
import pandas as pd

def parse_date_and_time(date_str, time_str):
    return datetime.strptime(f'{date_str} {time_str}', '%d/%m/%Y %H:%M:%S')

def get_x(item):
    return item['x']

def fetch_mag(slug, station_source):
    data = []
    tgo_password = os.environ.get('TGO_PASSWORD')
    if not tgo_password:
        raise ValueError("TGO_PASSWORD environment variable is not set")

    if station_source == 'tgo':
        response = requests.get(f'https://flux.phys.uit.no/cgi-bin/mkascii.cgi?site={slug}&year=2021&month=1&day=1&res=1min&pwd={tgo_password}&format=html&comps=DHZ&RTData=+Get+Realtime+Data+')
        lines = response.text.split('\n')

        for line in lines[7:-2]:
            line = line.strip()
            parts = line.split()
            x = float(parts[3])
            z = float(parts[4])
            y = float(parts[2])

            if x == 99999.9:
                continue

            data.append({
                'date': parse_date_and_time(parts[0], parts[1]),
                'X': x,
                'Y': y,
                'Z': z
            })

    elif station_source == 'fmi':
        response = requests.get(f'https://space.fmi.fi/image/realtime/UT/{slug.upper()}/{slug.upper()}data_24.txt')
        lines = response.text.split('\n')

        for line in lines[2:]:
            line = line.strip()
            if line:
                parts = line.split()
                year, month, day, hour, minute, second = map(int, parts[:6])
                x, y, z = map(float, parts[6:9])

                if x == 99999.9:
                    continue

                date = datetime(year, month, day, hour, minute, second)
                data.append({
                    'date': date,
                    'X': x,
                    'Y': y,
                    'Z': z
                })

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df.set_index('date', inplace=True)

    # return df[df.index >= df.index.max() - pd.Timedelta(minutes=45)]
    return df
import requests
import os
from datetime import datetime

def parse_date_and_time(date_str, time_str):
    return datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H:%M:%S')

def get_x(item):
    return item['x']

def fetch_mag(slug, station_source):
    data = []
    tgo_password = os.environ.get('TGO_PASSWORD')
    if not tgo_password:
        raise ValueError("TGO_PASSWORD environment variable is not set")

    if station_source == 'tgo':
        response = requests.get(f'/data/tgo/mkascii.cgi?site={slug}2a&year=2021&month=1&day=1&res=1min&pwd={tgo_password}&format=html&comps=DHZ&RTData=+Get+Realtime+Data+')
        lines = response.text.split('\n')

        for line in lines[7:-2]:
            line = line.strip()
            parts = line.split()
            x = float(parts[3])

            if x == 99999.9:
                continue

            data.append({
                'date': parse_date_and_time(parts[0], parts[1]),
                'x': x,
            })

    elif station_source == 'fmi':
        response = requests.get(f'/data/fmi/{slug.upper()}/{slug.upper()}data_24.txt')
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
                    'x': x,
                    'y': y,
                    'z': z
                })

    mean_x = sum(get_x(item) for item in data) / len(data)
    data = [{'date': d['date'], 'x': d['x'] - mean_x, **d} for d in data]

    return data

# Example usage
slug = 'your_slug_here'
station_source = 'tgo' # or 'fmi'
tgo_password = 'your_password_here'
result = fetch_mag(slug, station_source, tgo_password)

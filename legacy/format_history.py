# https://space.fmi.fi/image/www/index.php?page=monthly
# https://space.fmi.fi/image/www_old/IAGA.html

import pandas as pd
from datetime import datetime
import os
import requests
from tqdm import tqdm
import plotly.express as px

def download_iata():
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    base_url = 'https://space.fmi.fi/image/www/data_download_month.php?random=0.6728002336789753&source=image_web_page&starttime='
    params = '&sample_rate=60&compress=undefined&stations=TAR_'
    # params = '&sample_rate=60&compress=undefined&stations=NAL_LYR_HOR_HOP_BJN_NOR_SOR_KEV_TRO_MAS_AND_KIL_IVA_ABK_MUO_KIR_RST_SOD_PEL_JCK_DON_RAN_RVK_LYC_OUJ_MEK_HAN_DOB_SOL_NUR_HAR_UPS_KAR_PPNTAR_'

    for year in tqdm(range(2019, 2023), desc="Downloading data for each year"):
      for month in tqdm(months, desc="Downloading data for each month"):
          response = requests.get(f'{base_url}{year}{month}{params}').json()
          url = response['iaga_tar']['url']
          os.system(f"wget {url} -O ./temp/iata/{year}-{month}.tar")
          os.system(f"tar -xzf ./temp/iata/{year}-{month}.tar -C ./temp/iata/")
          os.system(f"rm ./temp/iata/{year}-{month}.tar")
          os.system(f"gunzip ./temp/iata/*.gz")
          os.system(f"rm ./temp/iata/*.gz")

          for file in os.listdir('./temp/iata/'):
            if file.endswith('.iaga'):
              print(f"Reading {file}")
              read_iaga(f'./temp/iata/{file}')

          os.system(f"rm ./temp/iata/*.iaga")

def read_iaga(file_path):
  records = {}

  with open(file_path, 'r') as file:
    while True:
      record = file.read(1440)
      if not record:
          break
      
      # Extracting metadata
      station = record[12:15]

      year = int(record[48:52])
      month = int(record[52:54])
      day = int(record[54:56])
      hour = int(record[56:58])

      values = record[159:1419]
      values = [int(values[i:i+7].replace(' ', '+')) for i in range(0, len(values), 7)]

      data_points = []
      for i in range(60):
        minute = int(f"{i:02}")
        value = values[i*3:(i+1)*3]
        data_points.append({
          'UT': datetime(year, month, day, hour, minute),
          'X': value[0],
          'Y': value[1],
          'Z': value[2]
        })

      if station not in records:
        records[station] = []
      records[station].extend(data_points)
  
  for key in records:
    print(f"Processing {key}")
    df = pd.DataFrame(records[key])
    # df.set_index('Date', inplace=True)

    df = df[(df['X'] != 999999) & (df['Y'] != 999999) & (df['Z'] != 999999)]

    df['X'] = df['X'] / 10
    df['Y'] = df['Y'] / 10
    df['Z'] = df['Z'] / 10

    csv_file_path = f'./temp/{key}.csv'
    try:
      existing_data = pd.read_csv(csv_file_path)
      combined_data = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
      combined_data = df

    combined_data.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    download_iata()
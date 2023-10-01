import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def get_sw(start_date, end_date):
	one_day = timedelta(days=1)

	dates = []
	current_date = start_date
	while current_date <= end_date:
		dates.append(current_date.strftime('%Y%m%d'))
		current_date += one_day


	buf = []
	for i in tqdm(range(len(dates))):
		date = dates[i]
		df = pd.read_csv(f'https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/{date}_ace_mag_1m.txt', delim_whitespace=True, skiprows=23, names=[
			'YR', 'MO', 'DA', 'HHMM', 'Day', 'Seconds', 'S', 'Bx', 'By', 'Bz', 'Bt', 'Lat', 'Long',
		])
		df['timestamp'] = pd.to_datetime(df['YR'].astype(str) + df['MO'].astype(str).str.zfill(2) + df['DA'].astype(str).str.zfill(2) + df['HHMM'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
		df = df.drop(columns=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Seconds', 'S', 'Lat', 'Long'])
		df.set_index('timestamp', inplace=True)

		plasma_df = pd.read_csv(f'https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/{date}_ace_swepam_1m.txt', delim_whitespace=True, skiprows=23, names=[
			'YR', 'MO', 'DA', 'HHMM', 'Day', 'Seconds', 'S', 'Density', 'Speed', 'Temperature',
		])
		plasma_df['timestamp'] = pd.to_datetime(plasma_df['YR'].astype(str) + plasma_df['MO'].astype(str).str.zfill(2) + plasma_df['DA'].astype(str).str.zfill(2) + plasma_df['HHMM'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
		plasma_df = plasma_df.drop(columns=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Seconds', 'S'])
		plasma_df.set_index('timestamp', inplace=True)

		buf.append(pd.merge(df, plasma_df, left_index=True, right_index=True))

	return pd.concat(buf)

output = pd.concat([
	get_sw(datetime(2018, 10, 18), datetime(2019, 1, 1)),
	get_sw(datetime(2020, 11, 10), datetime(2021, 1, 1))
])
output = output[output['Bx'] != -999.9]
output = output[output['Density'] != -9999.9]

output.to_csv('sw.csv', index=True)
print(output)

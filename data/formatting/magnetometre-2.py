import pandas as pd
from datetime import timedelta
import tqdm

# Load the CSV data into a DataFrame
print('start read')
df = pd.read_csv('big_iaga_filtered.csv')  # replace 'filename.csv' with your actual filename

print('date')
df['UT'] = pd.to_datetime(df['UT'])  # make sure 'DateTime' is in datetime format
print('date finished')

new_df_list = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    print(index)
    xyz_str = row['XYZ']
    date = row['UT']
    for i in range(0, len(xyz_str), 21):  # incrementing by 21 because each set of XYZ is 21 characters long
        xyz_set = xyz_str[i:i+21]
        x = xyz_set[0:7].strip()
        y = xyz_set[7:14].strip()
        z = xyz_set[14:21].strip()
        new_df_list.append({'StationId': row['StationId'], 'UT': date, 'X': x, 'Y': y, 'Z': z})
        date += timedelta(minutes=1)

# Optionally, save the DataFrame back to a CSV file
df.to_csv('big_iaga_filtered-2.csv', index=False)
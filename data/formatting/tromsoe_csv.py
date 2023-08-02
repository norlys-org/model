from datetime import datetime
import os
import pandas as pd

def format_date(date_str):
    date_obj = datetime.strptime(str(date_str), "%Y%m%d%H%M%S")
    formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date

def list_files(directory):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files

# Use the function
files = list_files('../tromsoe_csv')

# Read each file in the list
df_list = []
for file in files:
    if file.endswith('.DS_Store'):
        continue

    df = pd.read_csv(file, encoding='latin1')  # adjust parameters as necessary
    df = df[df['dd'] < 30] # filter dusk and dawn maximum 30%
    df = df.drop(['ac', 'ab'], axis=1)
    df['UT'] = df['UT'].apply(format_date)
    if not df[df['cloudy'] > 50].empty:
        continue
    df_list.append(df)

# Concatenate all dataframes
big_df = pd.concat(df_list, ignore_index=True)

# Write the big dataframe to a new CSV file
big_df.to_csv('big_file.csv', index=False)
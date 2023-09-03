import os
import pandas as pd

# Define field widths and column names according to your data specification
field_widths = [4, 3, 2, 6, 5, 5, 23, 4, 2, 2, 2, 2, 2, 1, 4, 2, 1, 2, 1, 1, 85, 180*7, 7, 7, 7]
column_names = ["RecordLength", "OneRecord", "TypeOfData", "StationId", "Latitude", "Longitude", "Free",
                "Year", "Month", "Day", "Hour", "Minute", "Interval", "HowProduced", "FilterBreakpoint",
                "FilterSlope", "BaselineInfo", "BaselineChange", "Components", "CharacterDay",
                "Free2", "XYZ", "MeanX", "MeanY", "MeanZ"]

# Specify the directory where .iaga files are stored
directory = "../iaga/"

# List to hold data frames
dfs = []

# Process all .iaga files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".iaga"):
        with open(os.path.join(directory, filename), 'r') as f:
            content = f.read()

            # Split the content into records of 1440 characters
            records = [content[i:i+1440] for i in range(0, len(content), 1440)]

            # For each record, create a DataFrame
            for record in records:
                # Split the record into fields according to the defined widths
                fields = [record[i:i+w] for i, w in zip([sum(field_widths[:j]) for j in range(len(field_widths))], field_widths)]
                
                # Create a DataFrame from the fields
                df = pd.DataFrame([fields], columns=column_names)
                
                # Append the DataFrame to the list
                dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Save the final DataFrame to a CSV file
final_df.to_csv('big_iaga.csv', index=False)
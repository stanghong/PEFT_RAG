# %%
import json
import os
import glob
import pandas as pd
# %%
def read_jsons_to_dataframe(input_folder):
    dataframes = []  # List to store individual dataframes

    for json_file in glob.glob(os.path.join(input_folder, '*.json')):
        try:
            print(f"Processing {json_file}")
            with open(json_file, 'r') as file:
                data = json.load(file)
                # Convert JSON data to a DataFrame
                df = pd.DataFrame([data])
                dataframes.append(df)
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    # Concatenate all dataframes into a single dataframe
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()
# %%
# Example usage
input_folder = './input_json_file/'

# Process JSON files and get combined dataframe
combined_df = read_jsons_to_dataframe(input_folder)

# Display the combined dataframe
print(combined_df)

# %%
combined_df.isnull().sum()

# %%
combined_df[~combined_df['person_notes'].isnull()]['person_notes']

# %%
# !pip install dtale
# %%
import dtale
# %%
d = dtale.show(combined_df[~combined_df['person_notes'].isnull()]['person_notes'])
d

# %%

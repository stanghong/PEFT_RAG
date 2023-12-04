# %%
import json
import os
import glob
import csv
# %%
def json_to_csv(json_file_path, output_folder):
    file_name = os.path.splitext(os.path.basename(json_file_path))[0]
    csv_file_path = os.path.join(output_folder, f"{file_name}.csv")

    with open(json_file_path, 'r') as json_file:
        content = json.load(json_file)

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Assuming the JSON is a dictionary, write headers (keys)
        writer.writerow(content.keys())
        # Write the values
        writer.writerow(content.values())

    print(f"Saved CSV file at {csv_file_path}")
# %%
def process_jsons_in_folder(input_folder, output_folder):
    print(f"Processing JSON files in {input_folder}")
    os.makedirs(output_folder, exist_ok=True)
    print(input_folder, glob.glob(os.path.join(input_folder, '*.json')))
    for json_file in glob.glob(os.path.join(input_folder, '*.json')):
        print(f"Processing {json_file}")
        json_to_csv(json_file, output_folder)

# %%

process_jsons_in_folder(input_folder, output_folder)

print(f"CSV files saved in {output_folder}")

# %%

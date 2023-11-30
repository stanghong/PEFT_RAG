# %%
# !pip install PyPDF2

# %%
import PyPDF2
import json
import os
import glob
# %%
def pdf_to_json(pdf_file_path):
    file_name = os.path.splitext(os.path.basename(pdf_file_path))[0]  # Extract base name without extension

    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        text_content = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text_content += page.extract_text()

    return {file_name: text_content}

def save_json_to_folder(json_data, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for file_name, content in json_data.items():
        json_file_path = os.path.join(folder_path, f"{file_name}.json")
        with open(json_file_path, 'w') as json_file:
            json.dump({file_name: content}, json_file)  # Include the file name in the JSON structure


def process_pdfs_in_folder(input_folder, output_folder):
    for pdf_file in glob.glob(os.path.join(input_folder, '*.pdf')):
        print(f'processing{pdf_file}')
        json_data = pdf_to_json(pdf_file)
        print(json_data)
        save_json_to_folder(json_data, output_folder)
# %%
input_folder = './data'
output_folder = './output_json_files'
process_pdfs_in_folder(input_folder, output_folder)

print(f"JSON files saved in {output_folder}")

# %%
# !pwd
# %%

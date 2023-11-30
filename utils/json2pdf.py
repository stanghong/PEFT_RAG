# write json to pdf code
# %%
!pip install fpdf

# %%
import json
import os
import glob
from fpdf import FPDF
# %%
def clean_text(text):
    '''
    Replace Non-Latin Characters
    '''
    return text.encode('latin-1', 'replace').decode('latin-1')

def json_to_pdf(json_file_path, output_folder):
    file_name = os.path.splitext(os.path.basename(json_file_path))[0]
    pdf_file_path = os.path.join(output_folder, f"{file_name}.pdf")

    with open(json_file_path, 'r') as json_file:
        content = json.load(json_file)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for key, value in content.items():
        text = f"{key}: {value}"
        # pdf.multi_cell(0, 10, text) #content[file_name]
        pdf.multi_cell(0, 10, clean_text(text))

    os.makedirs(output_folder, exist_ok=True)
    pdf.output(pdf_file_path)

def process_jsons_in_folder(input_folder, output_folder):
    print(f'{input_folder}')
    for json_file in glob.glob(os.path.join(input_folder, '*.json')):
        print(f"Processing {json_file}")

        json_to_pdf(json_file, output_folder)
# %%
# Example usage
input_folder = './data_json'
output_folder = './output_pdfs'
process_jsons_in_folder(input_folder, output_folder)

print(f"PDF files saved in {output_folder}")
# %%

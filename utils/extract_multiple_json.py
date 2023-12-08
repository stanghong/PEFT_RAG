# %%
import json
# %%
# Function to extract content from the JSON data
def extract_content(json_data, file_name):
    # Extract the name
    name = json_data.get('name', 'No name provided')

    # Extract contact information
    contact_info = json_data.get('contact_information', {})
    phone_number = contact_info.get('phone_number', 'No phone number provided')
    email_address = contact_info.get('email_address', 'No email provided')
    linkedin_url = contact_info.get('linkedin_url', 'No LinkedIn URL provided')

    person_notes = json_data.get('person_notes', [])
    
    extracted_notes = ""
    for note in person_notes:
        extracted_notes += note["content"] + " "
    return {
        'uid': file_name,
        # 'name': name,
        # 'phone_number': phone_number,
        # 'email_address': email_address,
        # 'linkedin_url': linkedin_url,
        'extracted_notes': extracted_notes
    }
# %%
# # Extract the content
directory_path = 'input_json_file'
all_extracted_info = []
not_extracted_uid=[]

# Check if the file is a JSON file
for file_name in os.listdir(directory_path):
    if file_name.endswith('.json'):
        # Construct the full file path
        file_path = os.path.join(directory_path, file_name)
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     json_data = json.load(file)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            print(file_name)
            extracted_result = extract_content(json_data,file_name)
            all_extracted_info.append(extracted_result)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            not_extracted_uid.append(file_name)
            continue
# %%
with open('output.json', 'w') as json_file:
    json.dump(all_extracted_info, json_file)
# %%

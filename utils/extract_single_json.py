# %%
import json
# %%
# Load JSON data from a file
with open('./input_json_file/11ed879a-3ca7-1a47-a5d6-069b8b3570ef.json', 'r') as file:
    json_data = json.load(file)
# %%
# Function to extract content from the JSON data
def extract_content(json_data):
    # Extract the name
    name = json_data.get('name', 'No name provided')

    # Extract contact information
    contact_info = json_data.get('contact_information', {})
    phone_number = contact_info.get('phone_number', 'No phone number provided')
    email_address = contact_info.get('email_address', 'No email provided')
    linkedin_url = contact_info.get('linkedin_url', 'No LinkedIn URL provided')

    # Extract the first entry from person_notes
    # person_notes = json_data.get('person_notes', [])
    person_notes = json_data.get('person_notes', [])
    
    # first_note_content = person_notes[0].get('content') if person_notes else 'No notes available'
    extracted_notes = ""
    for note in person_notes:
        extracted_notes += note["content"] + " "
    return {
        'name': name,
        'phone_number': phone_number,
        'email_address': email_address,
        'linkedin_url': linkedin_url,
        'extracted_notes': extracted_notes
    }
# %%
# Extract the content
extracted_result = extract_content(json_data)
# %%
extracted_result
# %%

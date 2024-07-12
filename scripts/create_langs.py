import requests
from bs4 import BeautifulSoup
import langcodes
import re

# Define the URL and the path to the output file
url = 'https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html'
output_file_path = 'support_list_with_ietf.txt'

# Download the HTML file
response = requests.get(url)
response.raise_for_status()  # Ensure we got a valid response

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the ISO codes and language names
data = []
for p in soup.find_all('p'):
    text = p.get_text(strip=True)
    if text and 'Iso Code' not in text:
        iso_code, language_name = text.split(' ')  # Note: ' ' is an en space
        data.append((iso_code.strip(), language_name.strip()))

# Sanitize the ISO code to be ASCII and split if necessary
def sanitize_iso_code(iso_code):
    # Replace non-ASCII characters
    iso_code = re.sub(r'[^a-zA-Z0-9-_]', '', iso_code)
    return iso_code.split('-')

# Define a function to lookup IETF code using langcodes library
def lookup_ietf_code(iso_code_parts):
    try:
        base_code = iso_code_parts[0]
        dialect_code = ''
        if len(iso_code_parts) > 1 and iso_code_parts[1] == 'dialect':
            dialect_code = iso_code_parts[2]
            ietf_code = f"{base_code}-{dialect_code}"
        else:
            ietf_code = langcodes.Language.get(base_code).to_tag()
        return ietf_code
    except Exception as e:
        print(f"Error looking up IETF code for {'-'.join(iso_code_parts)}: {e}")
        return '-'.join(iso_code_parts)  # Fallback to original ISO parts if lookup fails

# Extract script information if present
def extract_script(iso_code_parts):
    if 'script_' in iso_code_parts:
        script_index = iso_code_parts.index('script_') + 1
        return iso_code_parts[script_index] if script_index < len(iso_code_parts) else 'Latin'
    return 'Latin'

# Write the data to a TSV file
with open(output_file_path, 'w') as output_file:
    # Write the header
    output_file.write('Iso Code\tLanguage Name\tIETF Code\tScript\n')
    
    # Write each row of data
    for iso_code, language_name in data:
        iso_code_parts = sanitize_iso_code(iso_code)
        ietf_code = lookup_ietf_code(iso_code_parts)
        script = extract_script(iso_code_parts)
        output_file.write(f'{iso_code}\t{language_name}\t{ietf_code}\t{script}\n')

print(f'Data has been written to {output_file_path}')

import re

# Define the file path
file_path = 'tools/coco2odvg.py'

# Define the new values according to the dataset
new_id_map = '{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}'# 7 classes of AquariumDataset
new_ori_map = '{"1": "fish", "2": "jellyfish", "3": "penguins", "4": "sharks", "5": "puffins", "6":"stingrays", "7": "starfish"}'

# Read the content of the file
with open(file_path, 'r') as file:
    content = file.read()

# Replace the id_map value using regex
content = re.sub(r'id_map\s*=\s*\{[^\}]*\}', f'id_map = {new_id_map}', content)

# Replace the ori_map value using regex
content = re.sub(r'ori_map\s*=\s*\{[^\}]*\}', f'ori_map = {new_ori_map}', content)

# Write the updated content back to the file
with open(file_path, 'w') as file:
    file.write(content)

print(f"Updated {file_path} successfully.")

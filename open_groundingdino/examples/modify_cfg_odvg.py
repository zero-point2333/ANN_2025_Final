import re

def modify_file(file_path):
    label_list_content = 'label_list = ["fish","jellyfish","penguin","puffin","shark","starfish","stingray"]\n'

    # Read the entire content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace use_coco_eval =TRUE with use_coco_eval =FALSE using regex
    content = re.sub(r'use_coco_eval\s*=\s*True', 'use_coco_eval = False', content)

    # Insert label_list after use_coco_eval = FALSE using regex
    content = re.sub(r'use_coco_eval\s*=\s*False', r'use_coco_eval = False\n\n' + label_list_content, content, count=1, flags=re.MULTILINE)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

# Paths to the files
cfg_coco_path = 'config/cfg_coco.py'
cfg_odvg_path = 'config/cfg_odvg.py'

# Modify both files
modify_file(cfg_coco_path)
modify_file(cfg_odvg_path)

print("Updated use_coco_eval to FALSE and added label_list using regex in both files.")
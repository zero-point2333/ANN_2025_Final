import json

# Define the content for the JSON file
content = {
    "0": "fish",
    "1": "jellyfish",
    "2": "penguin",
    "3": "puffin",
    "4": "shark",
    "5": "starfish",
    "6": "stingray",
}

# Define the file path
file_path = 'content/input_params/label.json'

# Write the content to the JSON file
with open(file_path, 'w') as file:
    json.dump(content, file)

print(f"File '{file_path}' created successfully.")

# Define the data
data = {
    "train": [
        {
            "root": "content/aquarium_data/train",#Train images
            "anno": "content/input_params/train.jsonl",#Odvg jsonl file
            "label_map": "content/input_params/label.json",# label.json file
            "dataset_mode": "odvg"
        }
    ],
    "val": [
        {
            "root": "content/aquarium_data/test",#Test Images
            "anno": "content/aquarium_data/test/_annotations.coco.json",#Test data Annotation file in COCO
            "label_map": None,
            "dataset_mode": "coco"
        }
    ]
}

file_path = 'config/datasets_mixed_odvg.json'

with open(file_path, 'w') as file:
    json.dump(data, file, indent=2)

print(f"Data has been written to {file_path}")
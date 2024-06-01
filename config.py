import json

def json_to_python(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

file_path = 'config/build/model.json'
config = json_to_python(file_path)
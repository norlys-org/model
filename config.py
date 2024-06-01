import json

def json_to_python(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

model_path = 'config/build/model.json'
kv_path = 'config/build/kv.json'

config = json_to_python(model_path)
config.update(json_to_python(kv_path))
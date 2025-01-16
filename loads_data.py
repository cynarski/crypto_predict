import json

def load_json_data(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data
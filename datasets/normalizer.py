import json

# Change this to desired source and destination files.
name = '.\original_datasets\mnist_train.json'
new_file = 'normalized_mnist_train.json'

def get_data(filename):
    f = open(filename, "r")
    contents = f.read()
    f.close()
    contents = json.loads(contents)
    return contents

def set_data(filename, data):
    out_file = open(filename, "w") 
    json.dump(data, out_file, separators = (',', ':')) 
    out_file.close()

def normalize(data_set, max_value):
    for data in data_set['data']:
        data['columns'] = [float(n) / max_value for n in data['columns']]

def find_max(data):
    max_val = 0.0
    for item in data['data']:
        if max(item['columns']) > max_val:
            max_val = max(item['columns'])
    return max_val

def main():
    data = get_data(name)
    max_value = find_max(data)
    normalize(data, max_value)
    set_data(new_file, data)

if __name__ == '__main__':
    main()
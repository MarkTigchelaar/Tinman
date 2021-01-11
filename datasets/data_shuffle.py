import random
import json

filename = 'normalized_mnist_train.json'

def main():
    data = get_data(filename)
    for i in range(len(data['data'])):
        rand_index = random.randint(0, len(data['data']) - 1)
        temp = data['data'][i]
        data['data'][i] = data['data'][rand_index]
        data['data'][rand_index] = temp
    set_data(filename, data)

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

if __name__ == '__main__':
    main()
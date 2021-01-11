import json

def main():
    # Alter these to desired settings
    new_file = 'setosa_versicolor_iris.json'
    data = get_data('normalized_iris.json')
    desired_classes = ['Iris_setosa', 'Iris_versicolor']
    desired_attributes = ['sepal_length', 'petal_length']


    class_indicies = []
    label = []
    for i in range(len(data['result_map'])):
        if data['result_map'][i] in desired_classes:
            class_indicies.append(i)
        else:
            label.append(i)
    cols = []
    for i in range(len(data['table_info']['column_names'])):
        if data['table_info']['column_names'][i] in desired_attributes:
            cols.append(i)


    altered_data = dict()
    change_attr = dict()
    change_attr['result_map'] = class_indicies
    change_attr['column_names'] = cols
    change_attr['columns'] = [i for i in cols]
    change_attr['label'] = label
    altered_data = copy_value(data, change_attr)

    set_data(new_file, altered_data)

    


def copy_value(value, change_attr):
    new_val = None
    if type(value) is dict:
        new_val = dict()
        for key in value.keys():
            if key in change_attr:
                if key == 'label' and value[key] in change_attr[key]:
                        return
                new_val[key] = copy_value(value[key], change_attr[key])
            else:
                new_val[key] = copy_value(value[key], change_attr)
        return new_val
    elif type(value) is str:
        return value
    elif type(value) is int or type(value) is float:
        return value
    elif type(value) is list:
        new_lst = []
        for idx in range(len(value)):
            if idx in change_attr:
                new_lst.append(value[idx])
            elif type(value[idx]) is dict:
                new_val = copy_value(value[idx], change_attr)
                if new_val is not None:
                    new_lst.append(new_val)
        return new_lst

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
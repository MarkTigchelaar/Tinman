import sys
import subprocess
import time
import json

def main():
    options = sys.argv
    executeable = '.\\target\\debug\\Tinman.exe'
    testfiles = get_nnet_tests()
    path = 'D:\\Projects\\Rust-Projects\\Tinman\\testing_files\\'
    for filename in testfiles:
        print('\ntest batch: \"' + filename.replace('.json', '\"'))
        start = time.time()
        subprocess.call([executeable, path + filename])
        stop = time.time()
        print('test \"' + filename.replace('.json', '\"') + ' took ' + str(stop - start) + '\n')

def get_data(filename):
    f = open(filename, "r")
    contents = f.read()
    f.close()
    contents = contents.split("\n")
    return contents

def get_nnet_tests():
    return [
        #'nnet_iris_tests.json',
        #'nnet_trainer_iris_test.json',
        #'breeder_test.json',
        'classifier_tests.json'
    ]

if __name__ == '__main__':
    main()